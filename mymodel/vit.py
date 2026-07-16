"""
Vision Transformer (ViT) — image classification backbone.

Reference: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.

Data flow:
    image [B, C, H, W]
      → patch embed (+ pos embed) + [CLS]
      → TransformerEncoder × L
      → CLS token → linear head → logits [B, num_classes]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class FeedForward(nn.Module):
    """MLP block: Linear → GELU → Linear."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block: x + Attn(LN(x)), x + MLP(LN(x))."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to tokens.

    Uses a Conv2d with kernel=stride=patch_size — equivalent to linear
    projection of flattened patches, but faster.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            tokens [B, N, D]  where N = (H/p)*(W/p)
        """
        x = self.proj(x)  # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x


class ViT(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, hidden_size))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image → CLS feature [B, D]."""
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, D]
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # CLS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            logits [B, num_classes]
        """
        return self.head(self.forward_features(x))


def _smoke() -> None:
    model = ViT(
        img_size=32,
        patch_size=8,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        num_classes=10,
    )
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10), y.shape
    print(f"ViT OK  out={tuple(y.shape)}  params={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    _smoke()
