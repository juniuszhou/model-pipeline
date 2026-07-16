"""Quantization for Transformers.

Quantization reduces memory and computation by representing floating-point numbers
with lower-precision integers (e.g. int8, int4). This module demonstrates the core
algorithms used in LLM quantization.

Key concepts:
  - Scale (s): maps float range to integer range
  - Zero-point (z): shifts the integer range to align with float range
  - Quantize: x_q = clamp(round(x / s) + z, q_min, q_max)
  - Dequantize: x_hat = (x_q - z) * s

Two common schemes:
  - Asymmetric (min-max): uses full [q_min, q_max] range, handles non-symmetric distributions
  - Symmetric (absmax): z=0, range is symmetric around zero, simpler for matrix math

Reference:
  - GPTQ: https://arxiv.org/abs/2210.17323
  - SmoothQuant: https://arxiv.org/abs/2211.10438
  - LLM.int8(): https://arxiv.org/abs/2208.07339
"""

from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1.  Quantization parameters: scale and zero-point
# ---------------------------------------------------------------------------


def asymmetric_scale_and_z(
    x: torch.Tensor,
    q_min: int = -128,
    q_max: int = 127,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute scale and zero-point for *asymmetric* (min-max) quantization.

    Algorithm (per-tensor over the whole input):
      1. Find x_min, x_max.
      2. scale = (x_max - x_min) / (q_max - q_min)
         This maps the float range [x_min, x_max] to the integer range [q_min, q_max].
      3. zero_point = round(q_min - x_min / scale)
         This shifts the integer range so that 0.0 in float maps near zero_point.
      4. Clamp zero_point to [q_min, q_max] to stay representable.

    Asymmetric quantization is useful when the tensor distribution is not
    symmetric (e.g., post-ReLU activations are non-negative).
    """
    x_min = x.min()
    x_max = x.max()

    scale = (x_max - x_min) / float(q_max - q_min)
    # Degenerate case: all values identical → avoid divide-by-zero.
    # Keep scale=1 and choose z so (round(x/s) + z) dequantizes back near x.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    zero_point = torch.round(q_min - x_min / scale)
    zero_point = torch.clamp(zero_point, q_min, q_max)
    # Integer-valued zp as float keeps autograd / broadcast math simple and
    # matches common observer implementations.
    return scale, zero_point


def symmetric_scale(
    x: torch.Tensor,
    q_max: int = 127,
) -> torch.Tensor:
    """Compute scale for *symmetric* (absmax) quantization.

    Algorithm:
      1. Find max(abs(x)).
      2. scale = max_abs / q_max
      3. zero_point = 0 (by definition)

    Symmetric quantization is simpler (no zero-point to track) and is the
    default choice for weight quantization in most LLM methods (GPTQ, AWQ, …).
    """
    max_abs = x.abs().max()
    scale = max_abs / float(q_max)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale


# ---------------------------------------------------------------------------
# 2.  Quantize and dequantize
# ---------------------------------------------------------------------------


def quantize_asymmetric(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    q_min: int = -128,
    q_max: int = 127,
) -> torch.Tensor:
    """Quantize float tensor to int8 using asymmetric scheme.

    x_q = clamp(round(x / s) + z, q_min, q_max)

    Args:
        x: Input float tensor.
        scale: Quantization scale (broadcastable).
        zero_point: Quantization zero-point (broadcastable).

    Returns:
        Integer tensor in [q_min, q_max] (dtype int8 for the default range).
    """
    x_q = torch.round(x / scale) + zero_point
    x_q = torch.clamp(x_q, q_min, q_max)
    return x_q.to(torch.int8)


def quantize_symmetric(
    x: torch.Tensor,
    scale: torch.Tensor,
    q_min: int = -128,
    q_max: int = 127,
) -> torch.Tensor:
    """Quantize float tensor to int8 using symmetric scheme.

    x_q = clamp(round(x / s), q_min, q_max)

    Symmetric: zero_point is always 0, so we skip the addition.
    Scale is usually max_abs/q_max, so values map into [-q_max, q_max]
    (the extra int8 code -128 is unused under pure absmax).
    """
    x_q = torch.round(x / scale)
    x_q = torch.clamp(x_q, q_min, q_max)
    return x_q.to(torch.int8)


def dequantize(
    x_q: Int[Tensor, "..."] | torch.Tensor,
    scale: Float[Tensor, "..."] | torch.Tensor,
    zero_point: torch.Tensor | None = None,
) -> Float[Tensor, "..."]:
    """Dequantize integer tensor back to float.

    x_hat = (x_q - z) * s

    For symmetric quantization, zero_point is None (treated as 0).
    """
    x_f = x_q.float()
    if zero_point is None:
        return x_f * scale
    return (x_f - zero_point.float()) * scale


# ---------------------------------------------------------------------------
# 3.  Quantized matrix multiplication (int8 matmul)
# ---------------------------------------------------------------------------


def quantized_matmul(
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Simulate int8 quantized matrix multiplication: activation @ weight.T

    Why quantize matmul?
      - The dominant cost in transformer inference is the matrix multiply
        in attention (QK^T, PV) and feed-forward layers.
      - Using int8 instead of fp16/fp32 reduces memory bandwidth by 2–4×
        and enables faster integer arithmetic on supported hardware.

    Algorithm:
      1. Quantize activation per-tensor (asymmetric) → A_q, s_a, z_a
      2. Quantize weight per-tensor (symmetric)     → W_q, s_w   (z_w = 0)
         Weights are typically quantized *symmetrically* because learned
         weight distributions are roughly symmetric around zero.
      3. Integer path (hardware): C_acc = (A_q - z_a) @ W_q^T   (int32 accum)
         then C = C_acc * (s_a * s_w)
      4. Educational path used here: dequantize then matmul in float.
         Mathematically equivalent to (3) for this affine scheme.

    Note:
      A real implementation would use int8 GEMM / custom CUDA kernels.
      We dequantize to float then multiply only to show the data flow.
    """
    s_a, z_a = asymmetric_scale_and_z(activation)
    s_w = symmetric_scale(weight)

    a_q = quantize_asymmetric(activation, s_a, z_a)
    w_q = quantize_symmetric(weight, s_w)

    # Hardware would accumulate in int32; we simulate with float dequant.
    a_dq = dequantize(a_q, s_a, z_a)
    w_dq = dequantize(w_q, s_w, zero_point=None)

    return a_dq @ w_dq.T


# ---------------------------------------------------------------------------
# 4.  Fake quantization (for Quantization-Aware Training)
# ---------------------------------------------------------------------------


class FakeQuantize(torch.autograd.Function):
    """Fake quantization with straight-through estimator (STE).

    What is fake quantization?
      During QAT (Quantization-Aware Training), we need to simulate the
      effect of quantization in the forward pass while keeping useful
      gradients in the backward pass.

    Forward:  quantize → dequantize (simulates information loss)
    Backward: STE — treat round as identity; optionally zero grad outside
              the representable float range induced by [q_min, q_max].

    STE (Straight-Through Estimator):
      round() has zero derivative almost everywhere. STE bypasses that by
      treating quantize–dequantize as identity (within range) in backward.

    Used in:
      - Google QAT (TensorFlow)
      - torch.ao.quantization FakeQuantize
      - NVIDIA TensorRT training-time quantization
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        q_min: int,
        q_max: int,
    ) -> torch.Tensor:
        # Ensure scale / zp live on the same device & dtype as x for CUDA safety.
        scale = scale.to(device=x.device, dtype=x.dtype)
        zero_point = zero_point.to(device=x.device, dtype=x.dtype)

        x_int = torch.round(x / scale) + zero_point
        x_q = torch.clamp(x_int, q_min, q_max)
        x_dq = (x_q - zero_point) * scale

        # Float range that maps into [q_min, q_max] (for STE masking).
        x_min = (q_min - zero_point) * scale
        x_max = (q_max - zero_point) * scale
        ctx.save_for_backward(x, x_min, x_max)
        return x_dq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, x_min, x_max = ctx.saved_tensors
        # Clamp-aware STE: no gradient where forward was saturated by clamp.
        mask = (x >= x_min) & (x <= x_max)
        grad_x = grad_output * mask.to(dtype=grad_output.dtype)
        return grad_x, None, None, None, None


class FakeQuantizedLinear(nn.Module):
    """Linear layer with fake quantization, simulating int8 inference.

    During training:
      - Weights and activations are fake-quantized in the forward pass.
      - Gradients flow through via STE, so the model learns to be
        robust to quantization error.

    During inference:
      - Weights *could* be stored as real int8 and matmul done with
        integer arithmetic; here we keep fake-quant for simplicity.

    This module demonstrates QAT: the model adapts so the real int8
    version after training suffers minimal accuracy loss.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.q_min = -128
        self.q_max = 127

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fake-quantize weights (symmetric → zp = 0 on same device as weight).
        w_scale = symmetric_scale(self.weight, self.q_max)
        zero = torch.zeros((), device=self.weight.device, dtype=self.weight.dtype)
        w_q = FakeQuantize.apply(self.weight, w_scale, zero, self.q_min, self.q_max)

        # Fake-quantize activations (asymmetric, per-tensor).
        a_scale, a_z = asymmetric_scale_and_z(x, self.q_min, self.q_max)
        x_q = FakeQuantize.apply(x, a_scale, a_z, self.q_min, self.q_max)

        out = x_q @ w_q.t()
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# 5.  Weight quantization helpers (for post-training quantization, PTQ)
# ---------------------------------------------------------------------------


def quantize_weights_per_channel(
    linear: nn.Linear,
    num_bits: int = 8,
) -> dict:
    """Quantize a Linear layer's weights per output-channel (symmetric absmax).

    Why per-channel?
      - Different output channels can have very different weight magnitudes.
      - Per-tensor quantization lets large-magnitude channels waste bins and
        destroy precision on smaller channels.
      - Per-channel gives each output channel its own scale.

    Used in:
      - LLM.int8() (Dettmers et al., 2022)
      - GPTQ (Frantar et al., 2022)
      - AWQ (Lin et al., 2023)

    Args:
        linear: A nn.Linear layer.
        num_bits: 8 → levels in [-128, 127]; 4 → levels in [-8, 7].
            Both are stored as ``torch.int8`` tensors. (Native ``torch.int4``
            still lacks a usable general storage/copy path for this demo.)

    Returns:
        dict with keys:
          - ``weight_q``: int8 tensor of quantized codes
          - ``scale``: float per-channel scale, shape [out_features]
          - ``zero_point``: zeros per-channel (symmetric)
          - ``num_bits``: 8 or 4
    """
    if num_bits not in (4, 8):
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    w = linear.weight.data
    q_max = 7 if num_bits == 4 else 127
    q_min = -q_max - 1  # 8-bit: -128; 4-bit levels: -8

    # Per-channel: each output channel (dim=0) gets its own scale.
    max_abs = w.abs().max(dim=1, keepdim=True).values
    scale = max_abs / float(q_max)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    w_q = torch.clamp(torch.round(w / scale), q_min, q_max)
    # Store codes as int8 (true int8, or 4-bit codes in an int8 container).
    storage = w_q.to(torch.int8)

    return {
        "weight_q": storage,
        "scale": scale.squeeze(1),
        "zero_point": torch.zeros(w.shape[0], dtype=torch.int8, device=w.device),
        "num_bits": num_bits,
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_quantization():
    """Run a quick demonstration of all quantization methods."""
    torch.manual_seed(42)

    x = torch.randn(4, 64)
    w = torch.randn(128, 64)

    print("=" * 60)
    print("Quantization Demo")
    print("=" * 60)

    # --- asymmetric ---
    s_a, z_a = asymmetric_scale_and_z(x)
    x_q_a = quantize_asymmetric(x, s_a, z_a)
    x_dq_a = dequantize(x_q_a, s_a, z_a)
    mae_a = (x - x_dq_a).abs().mean()
    print(f"Asymmetric  | scale={s_a.item():.6f}, zp={z_a.item():.0f}, MAE={mae_a:.6f}")

    # --- symmetric ---
    s_s = symmetric_scale(x)
    x_q_s = quantize_symmetric(x, s_s)
    x_dq_s = dequantize(x_q_s, s_s)
    mae_s = (x - x_dq_s).abs().mean()
    print(f"Symmetric   | scale={s_s.item():.6f}, zp=0, MAE={mae_s:.6f}")

    # --- quantized matmul ---
    result_ref = x @ w.t()
    result_q = quantized_matmul(x, w)
    mae_mm = (result_ref - result_q).abs().mean()
    print(
        f"QuantMatmul | MAE={mae_mm:.6f}, "
        f"relative_err={mae_mm / result_ref.abs().mean():.6f}"
    )

    # --- fake quantization forward pass ---
    fq = FakeQuantizedLinear(64, 128)
    out_fq = fq(x)
    print(f"FakeQuantLin| output shape={out_fq.shape}")

    # STE: in-range grads pass; saturated values get 0 grad
    x_ste = torch.tensor([0.0, 1000.0], requires_grad=True)
    s_ste = torch.tensor(1.0)
    z_ste = torch.tensor(0.0)
    y_ste = FakeQuantize.apply(x_ste, s_ste, z_ste, -128, 127)
    y_ste.sum().backward()
    print(
        f"STE grads   | in-range={x_ste.grad[0].item():.1f}, "
        f"saturated={x_ste.grad[1].item():.1f} (expect 1, 0)"
    )

    # --- per-channel weight quant ---
    linear = nn.Linear(64, 128)
    q_w = quantize_weights_per_channel(linear, num_bits=8)
    print(
        f"PerChannel  | weight_q shape={q_w['weight_q'].shape}, "
        f"scale shape={q_w['scale'].shape}, bits={q_w['num_bits']}"
    )
    q_w4 = quantize_weights_per_channel(linear, num_bits=4)
    print(
        f"PerChannel4 | weight_q shape={q_w4['weight_q'].shape}, "
        f"bits={q_w4['num_bits']}, "
        f"value range=[{q_w4['weight_q'].min().item()}, {q_w4['weight_q'].max().item()}]"
    )

    print("=" * 60)

    x = torch.tensor([0.5, -0.5, 1.0, -1.0, 2.0, -2.0])
    s, z = asymmetric_scale_and_z(x)
    x_q = quantize_asymmetric(x, s, z)
    x_dq = dequantize(x_q, s, z)
    print(f"x original:   {x}")
    print(f"x quantized:  {x_q}")
    print(f"x dequant:    {x_dq}")
    print(f"scale={s.item():.4f}, zero_point={z.item():.0f}")
    err = (x - x_dq).abs().max()
    print(f"max quantization error: {err:.4f}")


if __name__ == "__main__":
    demo_quantization()
