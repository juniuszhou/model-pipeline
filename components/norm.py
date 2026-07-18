"""Normalization helpers used across demos (RMSNorm, LayerNorm, adaLN).

────────────────────────────────────────────────────────────────────────
  Quick reference
────────────────────────────────────────────────────────────────────────

  RMSNorm (common in modern LLMs):
      y = x / sqrt(mean(x²) + ε)
      — no mean subtraction, only re-scale by root-mean-square.

  LayerNorm:
      y = (x − μ) / sqrt(σ² + ε)   [, then optional γ·y + β]
      — centre + re-scale over the last dimension.

  Adaptive LayerNorm (adaLN, used in DiT / StyleGAN-style conditioning):
      1) plain LayerNorm (no learnable γ, β)
      2) condition-dependent affine:  y = x_norm * (1 + scale) + shift
         where scale, shift are predicted from a condition vector c
         (e.g. timestep + text embedding).

  Common bug:  torch.Tensor.var defaults to unbiased=True (divide by n−1).
  Neural-net LayerNorm uses the population variance (divide by n), i.e.
  unbiased=False — same as torch.nn.functional.layer_norm.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
from jaxtyping import Float


def rmsnorm(
    x: Float[torch.Tensor, "..."],
    eps: float = 1e-6,
) -> Float[torch.Tensor, "..."]:
    """Root Mean Square Layer Normalization.

    Correct formula (Zhang & Sennrich, 2019):

        y = x / sqrt(mean(x²) + ε)

    Note: do **not** subtract the mean, and the mean is of *squares*,
    not of x itself.
    """
    # mean(x²) over the feature dim — NOT mean(x)
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + eps)


def layer_norm(
    x: Float[torch.Tensor, "..."],
    eps: float = 1e-5,
) -> Float[torch.Tensor, "..."]:
    """Standard LayerNorm over the last dimension (no learnable γ, β).

    Formula:

        μ  = mean(x)
        σ² = mean((x − μ)²)          # population variance (÷ n)
        y  = (x − μ) / sqrt(σ² + ε)

    Matches ``F.layer_norm(x, x.shape[-1:])`` (default weight/bias = None
    behaves like identity affine, equivalent when weight=1, bias=0).
    """
    mean = x.mean(dim=-1, keepdim=True)
    # unbiased=False → divide by n, matching nn.LayerNorm / F.layer_norm
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps)


def adaptive_LayerNorm(
    x: Float[torch.Tensor, "..."],
    shift: Float[torch.Tensor, "..."],
    scale: Float[torch.Tensor, "..."],
    eps: float = 1e-6,
) -> Float[torch.Tensor, "..."]:
    """Adaptive LayerNorm (adaLN) as used in DiT.

    Steps:
      1. Layer-normalize x over the last dim (no fixed γ/β).
      2. Apply condition-dependent affine:

            y = x_norm * (1 + scale) + shift

    ``shift`` and ``scale`` are produced from a conditioning signal
    (timestep, class, text, …), e.g. via a small Linear on c.

    Shape convention (DiT):
        x:     [B, N, D]  or  [B, D]  or any leading dims + feature dim D
        shift: [B, D]     (broadcast over sequence length N)
        scale: [B, D]

    The ``(1 + scale)`` form (instead of bare ``scale``) pairs with
    zero-init of the modulation head: at init scale≈0, shift≈0 →
    y ≈ x_norm, so the network starts near plain LayerNorm.
    """
    # ── 1. plain LayerNorm (centre + unit variance) ───────────────
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) * torch.rsqrt(var + eps)

    # ── 2. condition-dependent affine (broadcast shift/scale) ─────
    # If shift/scale are [B, D] and x is [B, N, D], unsqueeze the mid dim.
    if shift.ndim == x_norm.ndim - 1:
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)

    return x_norm * (1.0 + scale) + shift


def from_torch(
    x: Float[torch.Tensor, "..."],
) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]:
    """Reference outputs from PyTorch for comparison / debugging."""
    ln = torch.nn.functional.layer_norm(x, x.shape[-1:])
    rn = torch.nn.functional.rms_norm(x, x.shape[-1:])
    return ln, rn
