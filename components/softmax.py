from __future__ import annotations

import math

import torch
from jaxtyping import Float


def softmax(
    x: Float[torch.Tensor, "... last"], dim: int = -1
) -> Float[torch.Tensor, "... last"]:
    # keepdim=True to keep the shape of the tensor, but last dim is 1
    x_max: Float[torch.Tensor, "... 1"] = x.max(dim=dim, keepdim=True).values
    exp_x: Float[torch.Tensor, "... last"] = torch.exp(x - x_max)
    # the total probability of the tensor should be 1
    result: Float[torch.Tensor, "... last"] = exp_x / exp_x.sum(dim=dim, keepdim=True)
    return result


def sigmoid(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    # x = 0 -> 0.5
    # x = inf -> 1
    # x = -inf -> 0
    return 1 / (1 + torch.exp(-x))


def stable_sigmoid(
    x: Float[torch.Tensor, "... last"],
) -> Float[torch.Tensor, "... last"]:
    """prevent exp(-x) from overflowing"""
    x = torch.clip(x, -500, 500)
    return 1 / (1 + torch.exp(-x))


def relu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    # x < 0 -> 0
    # x >= 0 -> x
    return torch.max(torch.zeros_like(x), x)


def gelu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    # x < 0 -> 0
    # x >= 0 -> x
    return (
        0.5
        * x
        * (1 + torch.tanh(torch.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


def swiglu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return x * sigmoid(x)
