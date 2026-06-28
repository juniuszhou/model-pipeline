from __future__ import annotations

import torch
from jaxtyping import Float


# convert the probability to logit, logit = log(p / (1 - p))
def logit(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    # x = 0 -> -inf
    # x = 1 -> inf
    # x = 0.5 -> 0
    # avoid log(0) and log(1)
    x = torch.clip(x, 1e-7, 1 - 1e-7)
    return torch.log(x / (1 - x))
