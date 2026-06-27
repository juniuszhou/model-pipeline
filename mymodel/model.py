"""Transformer language model architecture implemented in this module.

High-level data flow (TransformerLM):

```text
token_ids [B, S]
      │
      ▼
┌─────────────┐
│  Embedding  │  vocab_size → d_model
└─────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  TransformerBlock  × num_layers                             │
│                                                             │
│   x ────────────────────────────────────────────────────┐   │
│   │                                                     │   │
│   ▼                                                     │   │
│ ┌───────────────────────────────────────┐               │   │
│ │ DecoderModel  (Multi-Head Self-Attn)  │               │   │
│ │                                       │               │   │
│ │  Q = q_proj(x)                        │               │   │
│ │  K = k_proj(x)                        │               │   │
│ │  V = v_proj(x)                        │               │   │
│ │       │ split into num_heads          │               │   │
│ │       ▼                               │               │   │
│ │  RoPE(Q), RoPE(K)                     │               │   │
│ │       │                               │               │   │
│ │       ▼                               │               │   │
│ │  Scaled Dot-Product Attention         │               │   │
│ │  + causal mask (lower-triangular)     │               │   │
│ │       │                               │               │   │
│ │       ▼                               │               │   │
│ │  merge heads → o_proj                 │               │   │
│ └───────────────────────────────────────┘               │   │
│   │                                                     │   │
│   └─────────────────────────────── (+) ─────────────────┘   │
│   x (residual)                                              │
│   │                                                         │
│   ▼                                                         │
│ ┌───────────────────────────────────────┐                   │
│ │ FeedForwardModel  (SwiGLU FFN)        │                   │
│ │                                       │                   │
│ │  gate = SiLU(gate_proj(x))            │                   │
│ │  up   = up_proj(x)                    │                   │
│ │  hidden = gate * up                   │                   │
│ │  output = down_proj(hidden)           │                   │
│ └───────────────────────────────────────┘                   │
│   │                                                         │
│   └─────────────────────────────── (+) ─────────────────────┘
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────┐
│   RMSNorm   │
└─────────────┘
      │
      ▼
┌─────────────┐
│   LM Head   │  d_model → vocab_size
└─────────────┘
      │
      ▼
logits [B, S, vocab_size]

PretrainModel is a separate wrapper around HuggingFace AutoModelForCausalLM.
"""

from __future__ import annotations

import json
import logging
import os

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int, jaxtyped
from safetensors.torch import load_file, save_file
from torch import Tensor

from projects.config import LLMTrainingConfig

MODEL_PATH = "models/"
logger = logging.getLogger(__name__)


def _model_config_dict(model: TransformerLM) -> dict:
    return {
        "model_type": "transformer_lm",
        "vocab_size": model.vocab_size,
        "context_length": model.context_length,
        "hidden_size": model.d_model,
        "num_hidden_layers": model.num_layers,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "rope_theta": model.rope_theta,
    }


def _config_from_checkpoint_dict(config: dict) -> LLMTrainingConfig:
    return LLMTrainingConfig(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        hidden_size=config.get("hidden_size", config.get("d_model")),
        num_hidden_layers=config.get("num_hidden_layers", config.get("num_layers")),
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    )


def save_model_safe(model: TransformerLM, name: str = "checkpoint") -> None:
    """Save weights in Hugging Face safetensors layout.

    Creates:
        models/{name}/model.safetensors
        models/{name}/config.json
    """
    output_dir = os.path.join(MODEL_PATH, name)
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "model.safetensors")
    config_path = os.path.join(output_dir, "config.json")

    save_file(model.state_dict(), weights_path)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(_model_config_dict(model), handle, indent=2)
    logger.info("Saved safetensors checkpoint to %s", output_dir)


def load_model_safe(
    name: str = "checkpoint",
    device: str | torch.device = "cpu",
) -> TransformerLM:
    output_dir = os.path.join(MODEL_PATH, name)
    weights_path = os.path.join(output_dir, "model.safetensors")
    config_path = os.path.join(output_dir, "config.json")

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.pop("model_type", None)

    model = TransformerLM(_config_from_checkpoint_dict(config))
    state_dict = load_file(weights_path, device=str(device))
    model.load_state_dict(state_dict)
    return model


def save_model(model: TransformerLM, name: str = "model.pth") -> None:
    os.makedirs(MODEL_PATH, exist_ok=True)
    path = os.path.join(MODEL_PATH, name)
    # just save the weights. depends on the pickle version and class name, pytorch if save whole model
    torch.save(model.state_dict(), path)
    # or save the weights and hyperparameters
    # torch.save({
    #     "model_state_dict": model.state_dict(),
    #     "hyperparameters": {
    #         "vocab_size": model.vocab_size,
    #         "context_length": model.context_length,
    #         "d_model": model.d_model,
    #         "num_layers": model.num_layers,
    #         "num_heads": model.num_heads,
    #         "d_ff": model.d_ff,
    #         "rope_theta": model.rope_theta,
    #     },
    # }, path)
    logger.info("Model saved to %s", path)


def load_model(model: TransformerLM, name: str = "model.pth") -> TransformerLM:
    path = os.path.join(MODEL_PATH, name)
    # load the weights only
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    # hyperparameters already in the model object
    # we must save hyperparameters before, if we need create TransformerLM object from saved data
    model.load_state_dict(state_dict)
    return model


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    # 计算 inverse frequency
    inv_freq: Float[Tensor, " d_half_k"] = theta ** (
        -torch.arange(
            0, d_k, 2, device=in_query_or_key.device, dtype=in_query_or_key.dtype
        )
        / d_k
    )
    theta_table: Float[Tensor, " max_seq_len d_half_k"] = torch.outer(
        torch.arange(
            max_seq_len, device=in_query_or_key.device, dtype=in_query_or_key.dtype
        ),
        inv_freq,
    )
    cos_pos: Float[Tensor, " max_seq_len d_k"] = theta_table.cos().repeat_interleave(
        2, dim=-1
    )
    sin_pos: Float[Tensor, " max_seq_len d_k"] = theta_table.sin().repeat_interleave(
        2, dim=-1
    )

    cos: Float[Tensor, " ... sequence_length d_k"] = cos_pos[token_positions]
    sin: Float[Tensor, " ... sequence_length d_k"] = sin_pos[token_positions]

    rotated_half: Float[Tensor, " ... sequence_length d_k"] = torch.stack(
        [-in_query_or_key[..., 1::2], in_query_or_key[..., ::2]], dim=-1
    ).flatten(start_dim=-2)

    return in_query_or_key * cos + rotated_half * sin


class DecoderModel(nn.Module):
    def __init__(self, num_heads: int, d_model: int, rope_theta: float):
        super().__init__()
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    @jaxtyped
    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        batch_size, seq_len, d_model = x.shape
        d_k = d_model // self.num_heads

        q: Float[Tensor, "batch seq head_dim"] = self.q_proj(x)
        k: Float[Tensor, "batch seq head_dim"] = self.k_proj(x)
        v: Float[Tensor, "batch seq head_dim"] = self.v_proj(x)

        token_positions = torch.arange(seq_len, device=x.device)

        causal_mask: Bool[Tensor, "seq seq"] = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        q = q.reshape(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.num_heads, d_k).transpose(1, 2)

        q = run_rope(d_k, self.rope_theta, seq_len, q, token_positions)
        k = run_rope(d_k, self.rope_theta, seq_len, k, token_positions)

        attn_scores = (
            q
            @ k.transpose(-2, -1)
            / torch.sqrt(torch.tensor(q.shape[-1], dtype=q.dtype, device=q.device))
        )
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        result = attn_weights @ v

        result = result.transpose(1, 2).reshape(batch_size, seq_len, d_model)

        result: Float[Tensor, "batch seq d_model"] = self.o_proj(result)

        return result


class FeedForwardModel(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.activation = nn.SiLU()

    # 对于forward的计算，由于在整个MLP中，它只要求最后一个维度是d_model，所以它可以接收任意shape的输入，只要最后一个维度是d_model
    # 这个是由broadcast机制决定的，Tensor可以灵活的进行shape的调整，来满足计算的需求。
    # 但是对于代码的理解会有点不直观
    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        gate: Float[Tensor, "batch seq d_ff"] = self.gate(x)
        activation: Float[Tensor, "batch seq d_ff"] = self.activation(gate)
        up: Float[Tensor, "batch seq d_ff"] = self.up(x)
        hidden: Float[Tensor, "batch seq d_ff"] = activation.mul(up)
        output: Float[Tensor, "batch seq d_model"] = self.down(hidden)
        return output


class MoEFeedForwardModel(nn.Module):
    """SwiGLU MoE feed-forward with token-wise sparse expert routing."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        moe_num: int,
        top_k: int = 1,
    ):
        super().__init__()
        if top_k < 1 or top_k > moe_num:
            raise ValueError("top_k must be between 1 and moe_num")
        self.moe_num = moe_num
        self.top_k = top_k
        self.gate = nn.Linear(d_model, moe_num, bias=False)
        self.experts = nn.ModuleList(
            FeedForwardModel(d_model, d_ff) for _ in range(moe_num)
        )

    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        batch_size, seq_len, d_model = x.shape
        router_logits: Float[Tensor, "batch seq moe_num"] = self.gate(x)
        router_probs: Float[Tensor, "batch seq moe_num"] = torch.softmax(
            router_logits, dim=-1
        )

        # ... should be batch * seq
        flat_x: Float[Tensor, "... d_model"] = x.reshape(-1, d_model)
        output: Float[Tensor, "... d_model"] = torch.zeros_like(flat_x)

        if self.top_k == 1:
            expert_indices: Int[Tensor, "..."] = router_probs.argmax(dim=-1).reshape(-1)
            flat_probs: Float[Tensor, "... moe_num"] = router_probs.reshape(
                -1, self.moe_num
            )
            for expert_id, expert in enumerate(self.experts):
                token_mask: Bool[Tensor, "..."] = expert_indices == expert_id
                if not token_mask.any():
                    continue
                # x 作为输入就会根据mask，来选择每个token是否要经过这个expert来计算。这样x会更加的稀疏
                x_to_expert: Float[Tensor, "... d_model"] = flat_x[token_mask]
                expert_out: Float[Tensor, "... d_model"] = expert(x_to_expert)
                weights = flat_probs[token_mask, expert_id].unsqueeze(-1)
                # 把输出也进行加权，这样就得到了每个token的输出
                output[token_mask] = expert_out * weights
        else:
            topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            flat_topk_probs = topk_probs.reshape(-1, self.top_k)
            flat_topk_indices = topk_indices.reshape(-1, self.top_k)
            # 对于每个topk，我们都要计算每个expert的输出，然后进行加权
            for k in range(self.top_k):
                for expert_id, expert in enumerate(self.experts):
                    token_mask = flat_topk_indices[:, k] == expert_id
                    if not token_mask.any():
                        continue
                    expert_out = expert(flat_x[token_mask])
                    weights = flat_topk_probs[token_mask, k].unsqueeze(-1)
                    output[token_mask] += expert_out * weights

        return output.reshape(batch_size, seq_len, d_model)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_ff: int,
        rope_theta: float,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.attention = DecoderModel(num_heads, d_model, rope_theta)
        self.feed_forward = FeedForwardModel(d_model, d_ff)
        self.attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        # norm + attention + residual
        attn_output = self.attention(self.norm1(x))
        attn_output = self.attention_dropout(attn_output)
        x = x + attn_output
        # norm + feed forward + residual
        ff_output = self.feed_forward(self.norm2(x))
        ff_output = self.feed_forward_dropout(ff_output)
        x = x + ff_output
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: LLMTrainingConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.context_length = config.context_length
        self.d_model = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_heads
        self.d_ff = config.d_ff
        self.rope_theta = config.rope_theta

        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                self.num_heads, self.d_model, self.d_ff, self.rope_theta, self.moe
            )
            for _ in range(self.num_layers)
        ])
        self.norm = nn.RMSNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: Int[Tensor, "batch seq"]):
        x: Float[Tensor, "batch seq d_model"] = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x)

        nx: Float[Tensor, "batch seq d_model"] = self.norm(x)

        logits: Float[Tensor, "batch seq vocab_size"] = self.lm_head(nx)

        return logits

    @torch.inference_mode()
    def generate(self, input_ids: Int[Tensor, "batch seq"], max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits = self(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1)
        return input_ids
