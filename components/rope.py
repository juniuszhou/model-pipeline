from __future__ import annotations

import torch
from jaxtyping import Float, Int, Tensor


def rope(
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
    # calculate inverse frequency
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
