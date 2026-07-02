from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor


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
    # calculate inverse frequency, each position in half of the embedding dimension is a different frequency
    inv_freq: Float[Tensor, " d_half_k"] = theta ** (-torch.arange(0, d_k, 2) / d_k)

    # get a vector according to the position, then outer product with the inverse frequency to get a table of cosine and sine values
    theta_table: Float[Tensor, " max_seq_len d_half_k"] = torch.outer(
        torch.arange(max_seq_len),
        inv_freq,
    )

    # comppute cos and sin, then expand it to the full embedding dimension
    cos_pos: Float[Tensor, " max_seq_len d_k"] = theta_table.cos().repeat_interleave(
        2, dim=-1
    )

    sin_pos: Float[Tensor, " max_seq_len d_k"] = theta_table.sin().repeat_interleave(
        2, dim=-1
    )

    # cut the extra dimension off the token positions

    cos: Float[Tensor, " ... sequence_length d_k"] = cos_pos[token_positions]

    sin: Float[Tensor, " ... sequence_length d_k"] = sin_pos[token_positions]

    # rotate the half of the embedding dimension, it will be use to multiple with sin
    rotated_half: Float[Tensor, " ... sequence_length d_k"] = torch.stack(
        [-in_query_or_key[..., 1::2], in_query_or_key[..., ::2]], dim=-1
    ).flatten(start_dim=-2)
    return in_query_or_key * cos + rotated_half * sin


# def main():
#     d_k = 4
#     theta = 10000.0
#     max_seq_len = 4
#     in_query_or_key = torch.randn(1, 4, 4)
#     token_positions = torch.arange(4)
#     print(rope(d_k, theta, max_seq_len, in_query_or_key, token_positions))


# if __name__ == "__main__":
#     main()
