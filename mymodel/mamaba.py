"""
MAMABA: A simple Mamba (State Space Model) implementation for demonstration.

Key differences from Transformer:
  ┌──────────────────────────────────────────────────────────┬──────────────────────────────────┐
  │             Transformer                                  │          Mamba (SSM)             │
  ├──────────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ Self-attention: Q · K^T (quadratic O(S²·d))             │ Selective SSM scan (linear O(S)) │
  │ Positional encoding (RoPE, Sinusoidal, etc.)             │ No positional encoding needed    │
  │ Fixed context window                                     │ Infinite context (theoretically) │
  │ Separate Q/K/V projections                               │ Single projection → SSM params   │
  │ ReLU/SiLU/GELU activation in FFN                         │ SiLU activation only             │
  │ LayerNorm before each sublayer                           │ RMSNorm before each sublayer     │
  └──────────────────────────────────────────────────────────┴──────────────────────────────────┘

Mamba's core innovation:
  Instead of comparing every token to every other token (attention),
  Mamba compresses the entire sequence into a continuously updated
  hidden state h (the "state" in State Space Model).

  At each step t:
    h_t = Ā · h_{t-1} + B̄ · x_t    ← "write" to memory
    y_t = C · h_t + D · x_t           ← "read" from memory

  The parameters B, C, and Δ (which controls Ā, B̄) are computed
  FROM the input x, making the model "selective" about what to
  store and retrieve — this is what makes Mamba powerful.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor


class MambaBlock(nn.Module):
    """
    Core Mamba building block, replacing Transformer's self-attention + FFN.

    Data flow:
    ```
    x [B, S, d_model]
      │
      ▼
    ┌────────────────────────────────────────┐
    │  RMSNorm                                │
    └────────────────────────────────────────┘
      │
      ├──────────────────────┐
      ▼                      ▼
    ┌──────────────┐   ┌──────────┐
    │  proj (×2)   │   │  proj    │  ← z branch (gating)
    │  x → xz      │   │  x → z   │
    └──────┬───────┘   └────┬─────┘
           ▼                │
    ┌──────────────┐        │
    │   Conv1d     │        │
    │ (depthwise)  │        │
    └──────┬───────┘        │
           ▼                │
    ┌──────────────┐        │
    │   SiLU       │        │
    └──────┬───────┘        │
           ▼                │
    ┌──────────────┐        │
    │  Selective   │        │
    │   SSM       │        │
    └──────┬───────┘        │
           ▼                │
    ┌──────────────┐        │
    │  SiLU(z) ×   │◄───────┘  ← gating: multiply with activated z
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  out_proj    │
    └──────┬───────┘
           ▼
      output [B, S, d_model]  (+ residual from input)
    ```

    Differences from Transformer block:
    - Transformer has TWO sublayers per block (attention + FFN).
      MambaBlock has ONE sublayer that does both.
    - No multi-head split, no QKV, no RoPE, no causal mask.
    - Instead: 1D conv + SiLU + SSM + gating.
    - This single block has linear O(S) complexity in sequence length.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        """
        Args:
            d_model: Model/embedding dimension.
            d_state: SSM hidden state dimension (how much "memory" per channel).
                     This is like the hidden size of the recurrent state.
            d_conv:  Kernel size of the 1D convolution (local processing).
                     Think of this as replacing the "attention span" locally.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # ── Input normalization (like Pre-LN in Transformer) ──
        self.norm = nn.RMSNorm(d_model)

        # ── Gated input projection ──
        # Unlike Transformer which has separate Q, K, V projections,
        # Mamba projects x into two branches: xz and z.
        # xz: goes through conv + SSM path
        # z:  serves as a gate (like SwiGLU's gate in Transformer FFN)
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)

        # ── 1D Convolution ──
        # This is placed BEFORE the SSM to improve local processing.
        # There's no local processing in the SSM (it's global),
        # so the conv adds local pattern extraction.
        # Dimension: (d_model, d_conv) → kernel over timesteps
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal padding (no peeking ahead)
            groups=d_model,  # depthwise: each channel processed independently
        )

        # ── SSM parameters ──
        # log_A: The state transition matrix (stored in log form
        #        to ensure it stays negative → stable dynamics).
        # Shape: (d_model, d_state) — one diagonal matrix per input channel.
        # Unlike Transformer where each head has its own QK projection,
        # here each "channel" has its own A matrix.
        self.log_A = nn.Parameter(torch.randn(d_model, d_state) * 0.1)

        # D: The "skip connection" in the SSM output: y = C·h + D·x
        # This is analogous to the residual connection in Transformer.
        self.D = nn.Parameter(torch.ones(d_model))

        # Δ projection: computes discretization step from input.
        # Why Δ matters: it controls how much to "write" vs "keep".
        #   Δ large  → write aggressively (short-term focus)
        #   Δ small  → keep memory (long-term focus)
        # This is learned from the input, making the model selective.
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        # B and C projections: these map input → SSM input/output matrices.
        # This is the "selective" part — B and C vary per token,
        # unlike classic SSMs where they are fixed.
        # In the original Mamba, B and C share a projection from x.
        self.bc_proj = nn.Linear(d_model, 2 * d_state * d_model, bias=False)

        # ── Output projection ──
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Selective State Space Model (core of Mamba).

        Input:  x [B, S, d_model]   (after conv + SiLU)
        Output: y [B, S, d_model]

        The SSM is defined by the continuous-time system:
          h'(t) = A · h(t) + B · x(t)   (state evolution)
          y(t)  = C · h(t) + D · x(t)   (output)

        To apply it to discrete tokens, we discretize it with step Δ:
          Ā = exp(Δ · A)                (discrete state transition)
          B̄ = (exp(Δ · A) - I) · A⁻¹ · B ≈ Δ · B  (1st order approx)

        Then the recurrence is:
          h_t = Ā · h_{t-1} + B̄ · x_t
          y_t = C · h_t + D · x_t

        This is a LINEAR recurrence (like RNN but no nonlinearities),
        which is why it can be computed efficiently via scan.

        Key insight (the "selective" part):
          A is static, but B, C, and Δ are functions of the input x.
          This means the model can selectively decide to:
          - "remember" (small Δ, small B̄): keep old state
          - "update"   (large Δ, large B̄): write new info
          - "retrieve" (large C): read from state
        """
        B, S, d = x.shape
        d_state: int = self.d_state

        # 1. Compute Δ, B, C from input
        # Δ controls discretization (how much to write vs remember)
        # softplus ensures Δ > 0 (positive step size)
        dt = F.softplus(self.dt_proj(x))  # [B, S, d_model]

        # B: input → state mapping  (which parts of x enter state)
        # C: state → output mapping (which parts of state to read)
        # We split the projection into two halves: B and C
        bc = self.bc_proj(x)  # [B, S, 2*d_state*d_model]
        bc = bc.view(B, S, 2, d_state, d)  # [B, S, 2, d_state, d]
        B_proj = bc[:, :, 0]  # [B, S, d_state, d]
        C_proj = bc[:, :, 1]  # [B, S, d_state, d]

        # 2. Discretize: compute Ā and B̄
        # A is stored in log form for stability during training.
        # Ā = exp(Δ · A) — each channel shares the same A matrix.
        A = -torch.exp(self.log_A)  # [d_model, d_state], always negative
        A = A.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model, d_state]

        # Δ shape: [B, S, d_model] → [B, S, d_model, 1]
        dt = dt.unsqueeze(-1)  # [B, S, d_model, 1]
        A_bar = torch.exp(dt * A)  # [B, S, d_model, d_state]

        # B̄ ≈ Δ · B (zero-order hold approximation)
        # B shape: [B, S, d_state, d] — careful with dimensions
        B_proj = B_proj.transpose(-2, -1)  # [B, S, d, d_state]
        B_bar = dt * B_proj  # [B, S, d, d_state]

        # C shape: [B, S, d_state, d]
        C_proj = C_proj.transpose(-2, -1)  # [B, S, d, d_state]
        C = C_proj

        # 3. Recurrent scan (sequential for simplicity)
        # This is the O(S) step that replaces O(S²) attention.
        #
        # During inference this is just a simple for loop — fast.
        # During training the original Mamba uses an associative scan
        # to parallelize, but we keep the sequential version for clarity.
        h = torch.zeros(B, d, d_state, device=x.device, dtype=x.dtype)
        y: Float[Tensor, "batch seq d_model"] = torch.zeros_like(x)

        for t in range(S):
            # h_t = Ā_t · h_{t-1} + B̄_t · x_t
            # Ā: [B, d, d_state],  h: [B, d, d_state],  B̄: [B, d, d_state]
            # x: [B, d]
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # y_t = C_t · h_t + D · x_t
            # C: [B, d, d_state],  h: [B, d, d_state]
            # output: [B, d]
            y[:, t] = (C[:, t] * h).sum(dim=-1) + self.D * x[:, t]

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, d_model] input sequence
        Returns:
            [B, S, d_model] output sequence
        """
        residual = x
        x = self.norm(x)

        # Split into two branches (like SwiGLU gating in some Transformers)
        xz = self.in_proj(x)  # [B, S, 2*d_model]
        x, z = xz.chunk(2, dim=-1)  # each [B, S, d_model]

        # 1D Convolution (causal, no peeking ahead)
        # Conv1d expects [B, d_model, S], so permute
        # convulution 在二个不同的token直接做才有意义，如果在一个token内部做，则没有意义。因为它只是 把token的高维的信息混合在一起了。
        x = x.transpose(1, 2)  # [B, d_model, S]

        last_dim = x.shape[-1]
        x: Float[Tensor, "batch d_model (S - d_conv + 1)"] = self.conv1d(x)
        x = x[:, :, :last_dim]  # remove causal padding
        x = x.transpose(1, 2)  # [B, S, d_model]

        # Activation
        x = F.silu(x)

        # SSM
        x = self._ssm(x)  # [B, S, d_model]

        # Gate: like SwiGLU z * SiLU(x)
        z = F.silu(z)
        x = x * z

        # Output projection
        x = self.out_proj(x)

        return x + residual  # residual connection (like Transformer)


class MambaModel(nn.Module):
    """
    Stack of Mamba blocks (like Transformer's stacked decoder layers).
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)

    def forward(
        self, x: Float[Tensor, "batch seq d_model"]
    ) -> Float[Tensor, "batch seq d_model"]:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MambaForLM(nn.Module):
    """
    Full Mamba language model: Embedding → Mamba blocks → LM head.

    Compare with TransformerLM in model.py:
      ┌───────────────────────┬──────────────────────────┐
      │     TransformerLM     │     MambaForLM           │
      ├───────────────────────┼──────────────────────────┤
      │ Embedding             │ Embedding                │
      │ + PositionEmbedding   │ (no positional encoding) │
      │ TransformerBlock × L  │ MambaBlock × L           │
      │   ├─ Self-Attn O(S²)  │   ├─ Conv1d              │
      │   └─ FFN              │   └─ SSM scan O(S)       │
      │ RMSNorm               │ RMSNorm                  │
      │ LM Head               │ LM Head                  │
      └───────────────────────┴──────────────────────────┘
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # No positional encoding! The SSM recurrence inherently
        # models position through the sequential state updates.
        self.backbone = MambaModel(d_model, n_layers, d_state, d_conv)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embedding and lm_head weights (common in LLMs)
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, S] token indices
        Returns:
            logits: [B, S, vocab_size]
        """
        x = self.embed(input_ids)  # [B, S, d_model]
        x = self.backbone(x)  # [B, S, d_model]
        logits = self.lm_head(x)  # [B, S, vocab_size]
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation (one token at a time).

        Unlike Transformer generation (which re-processes the full prefix
        at each step or uses KV cache), Mamba generation is naturally
        efficient: we just continue the sequential scan from where we left off.
        However, for simplicity here we just run the full forward pass each time.

        Args:
            input_ids: [B, S] prompt tokens
            max_new_tokens: how many tokens to generate
            temperature: sampling temperature (lower = more deterministic)
        Returns:
            [B, S + max_new_tokens] generated sequence
        """
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids)  # [B, S_cur, vocab_size]
            next_logits = logits[:, -1, :]  # [B, vocab_size]

            if temperature > 0:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = next_logits.argmax(dim=-1, keepdim=True)

            input_ids = torch.cat([input_ids, next_id], dim=-1)

        return input_ids


# ═══════════════════════════════════════════════════════════
# Training code
# ═══════════════════════════════════════════════════════════


def make_synthetic_data(
    vocab_size: int,
    seq_len: int,
    num_samples: int,
    pattern_len: int = 8,
) -> list[Int[Tensor, "seq"]]:
    """
    Create a synthetic dataset with repeating patterns.
    The model needs to learn the pattern and predict the next token.

    Each sample: [0, 1, 2, ..., pattern_len-1, 0, 1, 2, ...]
    reshaped to seq_len. This is a simple next-token prediction task
    that verifies the SSM can capture sequential dependencies.
    """
    pattern = torch.arange(pattern_len) % vocab_size
    data = []
    for _ in range(num_samples):
        # Generate a long sequence by repeating the pattern
        full = pattern.repeat((seq_len // pattern_len) + 1)
        data.append(full[:seq_len].long())
    return data


def train_mamba(
    vocab_size: int = 16,
    d_model: int = 64,
    n_layers: int = 2,
    d_state: int = 8,
    d_conv: int = 4,
    seq_len: int = 32,
    batch_size: int = 8,
    lr: float = 1e-3,
    max_steps: int = 500,
    log_interval: int = 50,
    seed: int = 42,
) -> MambaForLM:
    """
    Train a Mamba model on a simple synthetic next-token prediction task.

    The task: learn a repeating numeric pattern (0,1,2,...,pattern_len-1).
    This is the simplest test of whether the SSM can capture sequence
    structure — the model must use its hidden state to know where it is
    in the pattern.

    Data flow per step:
    ```
    input_ids [B, S]           ← synthetic repeating pattern
      │
      ▼
    MambaForLM
      ├─ Embedding  (no pos encoding!)
      ├─ MambaBlock × n_layers  ← O(S) scan, no attention
      └─ LM Head
      │
      ▼
    logits [B, S, vocab_size]
      │
      ▼
    CrossEntropyLoss(logits[:-1], labels[1:])   ← next-token prediction
    """
    torch.manual_seed(seed)

    # ── Create synthetic data ──
    num_samples = batch_size * (max_steps + 1)
    data = make_synthetic_data(vocab_size, seq_len, num_samples)

    # ── Create model & optimizer ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaForLM(vocab_size, d_model, n_layers, d_state, d_conv).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training for {max_steps} steps...\n")

    model.train()
    step = 0
    sample_idx = 0

    while step < max_steps:
        # Build a batch from the synthetic data
        batch = data[sample_idx : sample_idx + batch_size]
        # update for next iteration
        sample_idx += batch_size
        input_ids: Int[Tensor, "batch seq"] = torch.stack(batch).to(device)  # [B, S]

        # Standard causal LM loss: predict next token at each position
        #   logits[:, :-1]  vs  labels[:, 1:]
        logits = model(input_ids)  # [B, S, vocab_size]
        shift_logits = logits[:, :-1, :].contiguous()  # [B, S-1, vocab_size]
        shift_labels = input_ids[:, 1:].contiguous()  # [B, S-1]

        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1

        if step % log_interval == 0 or step == 1:
            # Compute accuracy on this batch
            preds = shift_logits.argmax(dim=-1)
            acc = (preds == shift_labels).float().mean().item()
            print(f"step {step:>4} | loss {loss.item():.4f} | acc {acc:.3f}")

    print("\nTraining done!")
    return model


# ═══════════════════════════════════════════════════════════
# Demo: train and generate
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_mamba(
        vocab_size=16,
        d_model=64,
        n_layers=2,
        d_state=8,
        d_conv=4,
        seq_len=32,
        batch_size=8,
        lr=1e-3,
        max_steps=500,
    )

    # ── Test generation ──
    # Prompt the model with the start of the pattern [0, 1, 2]
    # and see if it correctly continues with 3, 4, 5, ...
    prompt = torch.tensor([[0, 1, 2]], device=device)
    output = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    print(f"\nPrompt:    {prompt[0].tolist()}")
    print(f"Generated: {output[0].tolist()}")
    # Expected: [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4]
    # The model learns the repeating pattern 0..7, so after 2 it should
    # generate 3, 4, 5, 6, 7, 0, 1, ...
