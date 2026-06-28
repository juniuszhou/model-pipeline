from __future__ import annotations

import torch
from jaxtyping import Float


# most of LLM models use rmsnorm now, the sub mean is not useful in trainingaccording to the paper
def rmsnorm(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    # x = x / sqrt(mean(x^2))
    mean = x.mean(dim=-1, keepdim=True)
    return x / torch.sqrt(mean)


# still useful if you need the data distribution around zero
def layer_norm(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    # x = x / sqrt(mean(x^2))
    mean = x.mean(dim=-1, keepdim=True)
    x = x - mean
    variance = x.var(dim=-1, keepdim=True)
    x = x / torch.sqrt(variance + 1e-5)
    return x


def from_torch(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    layer_norm = torch.nn.functional.layer_norm(x, x.shape[-1:])
    rms_norm = torch.nn.functional.rms_norm(x, x.shape[-1:])
    return layer_norm, rms_norm
