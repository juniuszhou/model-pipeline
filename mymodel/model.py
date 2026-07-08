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
try:
    from config import BlockConfig, LLMTrainingConfig, TransformerModelConfig
except ImportError:
    from mymodel.config import BlockConfig, LLMTrainingConfig, TransformerModelConfig
from jaxtyping import Bool, Float, Int, jaxtyped
from safetensors.torch import load_file, save_file
from torch import Tensor

MODEL_PATH = "models/"
logger = logging.getLogger(__name__)


def _model_config_dict(model: TransformerLM) -> dict:
    return {
        "model_type": "transformer_lm",
        "vocab_size": model.config.vocab_size,
        "context_length": model.config.context_length,
        "d_model": model.config.d_model,
        "num_hidden_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_heads,
        "d_ff": model.config.d_ff,
        "rope_theta": model.config.rope_theta,
    }


def _config_from_checkpoint_dict(config: dict) -> LLMTrainingConfig:
    return LLMTrainingConfig(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config.get("d_model", config.get("d_model")),
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
    name: str = "latest",
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
    # strict=False: tolerate extra keys from old checkpoints (e.g. removed unused norm)
    model.load_state_dict(state_dict, strict=False)
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


def load_model(model: TransformerLM, name: str = "latest") -> TransformerLM:
    path = os.path.join(MODEL_PATH, name)
    # load the weights only
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
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

    # x just required to be (... d_model), for the most LLM models, it is (batch seq d_model)
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
        # gate to different expert
        self.gate = nn.Linear(d_model, moe_num, bias=False)

        # each expert is a feed forward network
        self.experts = nn.ModuleList(
            FeedForwardModel(d_model, d_ff) for _ in range(moe_num)
        )

    def forward(self, x: Float[Tensor, "batch seq d_model"]):
        batch_size, seq_len, d_model = x.shape

        router_logits: Float[Tensor, "batch seq moe_num"] = self.gate(x)
        # each token go to which expert with probability
        router_probs: Float[Tensor, "batch seq moe_num"] = torch.softmax(
            router_logits, dim=-1
        )

        # 3 dim -> 2 dim, for each token, go to which expert
        flat_x: Float[Tensor, "batch-seq d_model"] = x.reshape(-1, d_model)
        output: Float[Tensor, "batch-seq d_model"] = torch.zeros_like(flat_x)

        if self.top_k == 1:
            # which expert with max probability, then each token go to which expert. the value is the expert index
            expert_indices: Int[Tensor, "batch seq"] = router_probs.argmax(dim=-1)

            flat_probs: Float[Tensor, "batch-seq moe_num"] = router_probs.reshape(
                -1, self.moe_num
            )
            for expert_id, expert in enumerate(self.experts):
                # 2 dim -> 1 dim, then we can use it to mask the input flat x
                token_mask: Bool[Tensor, "batch-seq"] = (
                    expert_indices == expert_id
                ).reshape(-1)

                if not token_mask.any():
                    continue
                # x as input to expert, should be sparse only part filter by gate to be invoked in expert computation.
                x_to_expert: Float[Tensor, "batch-seq d_model"] = flat_x[token_mask]

                # call normal feed forward model with the flat x
                expert_out: Float[Tensor, "batch-seq d_model"] = expert(
                    x_to_expert
                )  # batch-seq d_model

                # get the second dim according to the expert id, and the first dim according to the token mask
                weights: Float[Tensor, "batch-seq"] = flat_probs[token_mask, expert_id]

                # 1 dim -> 2 dim, then we can use it to multiply the expert output
                weights: Float[Tensor, "batch-seq 1"] = weights.unsqueeze(-1)

                output[token_mask] = expert_out * weights
        else:
            topk_probs: Float[Tensor, "batch seq top_k"]
            topk_indices: Int[Tensor, "batch seq top_k"]
            topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            flat_topk_probs: Float[Tensor, "batch-seq top_k"] = topk_probs.reshape(
                -1, self.top_k
            )
            flat_topk_indices: Int[Tensor, "batch-seq top_k"] = topk_indices.reshape(
                -1, self.top_k
            )
            # for each topk, we need to calculate the output of each expert, then add them together
            for k in range(self.top_k):
                for expert_id, expert in enumerate(self.experts):
                    token_mask: Bool[Tensor, "batch-seq"] = (
                        flat_topk_indices[:, k] == expert_id
                    )
                    if not token_mask.any():
                        continue
                    expert_out: Float[Tensor, "batch-seq d_model"] = expert(
                        flat_x[token_mask]
                    )
                    weights: Float[Tensor, "batch-seq 1"] = flat_topk_probs[
                        token_mask, k
                    ].unsqueeze(-1)
                    output[token_mask] += expert_out * weights

        return output.reshape(batch_size, seq_len, d_model)


class TransformerBlock(nn.Module):
    """
    Transformer block with attention and feed forward.
    step by step:
    1. norm x -> x
    2. attention x -> x
    3. dropout x -> x
    4. residual input + x * learning_rate -> x
    5. norm x -> x
    6. feed forward x -> x
    7. dropout x -> x
    8. residual x + x * learning_rate -> x
    return the output x
    """

    def __init__(
        self,
        config: BlockConfig,
    ):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.rope_theta = config.rope_theta
        self.use_moe = config.use_moe
        self.moe_num = config.moe_num
        self.top_k = config.top_k
        self.dropout = config.dropout
        self.learning_rate = config.learning_rate

        self.attention = DecoderModel(self.num_heads, self.d_model, self.rope_theta)
        if config.use_moe:
            self.feed_forward = MoEFeedForwardModel(
                self.d_model, self.d_ff, self.moe_num, self.top_k
            )
        else:
            self.feed_forward = FeedForwardModel(self.d_model, self.d_ff)

        self.attention_dropout = nn.Dropout(self.dropout)
        self.feed_forward_dropout = nn.Dropout(self.dropout)
        self.norm1 = nn.RMSNorm(self.d_model)
        self.norm2 = nn.RMSNorm(self.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq d_model"], learning_rate: float | None = None
    ):
        if learning_rate is None:
            learning_rate = self.learning_rate
        # norm + attention + residual
        attn_output = self.attention(self.norm1(x))
        attn_output = self.attention_dropout(attn_output)
        x = x + attn_output * self.learning_rate
        # norm + feed forward + residual
        ff_output = self.feed_forward(self.norm2(x))
        ff_output = self.feed_forward_dropout(ff_output)
        x = x + ff_output * self.learning_rate
        return x


class TransformerModel(nn.Module):
    """Transformer backbone: embedding → layers → norm."""

    def __init__(self, config: TransformerModelConfig):
        super().__init__()
        self.config = config

        block_config = BlockConfig(config)

        self.layers = nn.ModuleList([
            TransformerBlock(block_config) for _ in range(self.config.num_hidden_layers)
        ])
        self.norm = nn.RMSNorm(self.config.d_model)

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        learning_rate: float | None = None,
    ):
        for layer in self.layers:
            x = layer(x, learning_rate)
        return self.norm(x)


class TransformerLM(nn.Module):
    """Full language model: embedding → transformer → lm_head."""

    def __init__(self, config: LLMTrainingConfig):
        super().__init__()
        self.config = config

        model_config = TransformerModelConfig(config)
        self.model = TransformerModel(model_config)

        self.embed = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.lm_head = nn.Linear(
            self.config.d_model, self.config.vocab_size, bias=False
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # learning_rate is used to control the learning rate of the model, could be adaptive in different phases
    def forward(
        self, input_ids: Int[Tensor, "batch seq"], learning_rate: float | None = None
    ):
        x: Float[Tensor, "batch seq d_model"] = self.embed(input_ids)

        hidden_state: Float[Tensor, "batch seq d_model"] = self.model(x, learning_rate)
        logits: Float[Tensor, "batch seq vocab_size"] = self.lm_head(hidden_state)

        return logits

    @torch.inference_mode()
    # the input must be (batch seq), because the model is trained on (batch seq)
    def generate(self, input_ids: Int[Tensor, "batch seq"], max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits = self(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=-1)
        return input_ids
