"""
Video Vision Transformer (VideoViT)

Extends the image ViT (see vit.py) with a frame / time dimension.

────────────────────────────────────────────────────────────────────────
  Why not just flatten all frames into one long sequence?
────────────────────────────────────────────────────────────────────────

  Naive joint attention over all space-time tokens costs:

      O( (T · N)² · D )     T = #frames,  N = #spatial patches

  For T=16, N=196 (14×14), sequence length = 3136 → attention matrix
  is ~10M entries *per head per layer* — slow and memory-hungry.

  This file implements several **standard efficiency tricks** used in
  TimeSformer / ViViT / related video transformers.  Pick a strategy
  with ``attn_mode=...``:

  ┌──────────────────┬──────────────────────────────────────────────┐
  │ mode             │ idea & complexity (per layer, ignore heads)  │
  ├──────────────────┼──────────────────────────────────────────────┤
  │ space_only       │ encode each frame alone, pool over time      │
  │                  │ O(T · N²)  — fastest, no temporal mixing     │
  ├──────────────────┼──────────────────────────────────────────────┤
  │ divided          │ TimeSformer: temporal attn then spatial attn │
  │                  │ O(T²·N + T·N²)  — strong quality / cost      │
  ├──────────────────┼──────────────────────────────────────────────┤
  │ factorized       │ ViViT: full spatial encoder → temporal on    │
  │                  │ frame CLS tokens  O(T·N² + T²)               │
  ├──────────────────┼──────────────────────────────────────────────┤
  │ local_temporal   │ spatial full + temporal only in ±w window    │
  │                  │ O(T·N² + T·w·N)  — long videos stay linear   │
  ├──────────────────┼──────────────────────────────────────────────┤
  │ joint            │ full space-time attention (baseline only)    │
  │                  │ O((T·N)²)  — use ONLY for tiny T,N demos     │
  └──────────────────┴──────────────────────────────────────────────┘

  Extra embedding tricks:
    • **tubelet embedding** — 3D conv over (time, h, w) so each token
      already covers several frames → shorter sequence (ViViT-style).
    • separate **spatial** and **temporal** positional encodings
      (factorised pos-emb) so T can change without full re-init.

Run demo:
    python -m mymodel.video_vit
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import torch
import torch.nn as nn

try:
    from vit import FeedForward, MultiHeadSelfAttention, TransformerBlock
except ImportError:  # running as package: python -m mymodel.video_vit
    from mymodel.vit import FeedForward, MultiHeadSelfAttention, TransformerBlock

logger = logging.getLogger(__name__)

AttnMode = Literal[
    "space_only",
    "divided",
    "factorized",
    "local_temporal",
    "joint",
]


# ═══════════════════════════════════════════════════════════════════════
# 1. Embeddings
# ═══════════════════════════════════════════════════════════════════════


class TubeletEmbed(nn.Module):
    """3D patch / tubelet embedding (ViViT).

    A tubelet is a small space-time cube of size
    ``(tubelet_t, patch_size, patch_size)``.  One Conv3d with matching
    stride turns the whole video into tokens in a single pass — much
    cheaper than embedding every frame independently when tubelet_t > 1
    (sequence length shrinks by tubelet_t along time).

    Input:  [B, C, T, H, W]
    Output: [B, T', N, D]   where
            T' = T / tubelet_t
            N  = (H/p) * (W/p)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_t: int = 2,
        in_channels: int = 3,
        hidden_size: int = 768,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        assert num_frames % tubelet_t == 0
        self.patch_size = patch_size
        self.tubelet_t = tubelet_t
        self.grid_size = img_size // patch_size
        self.num_spatial = self.grid_size * self.grid_size
        self.num_temporal = num_frames // tubelet_t

        # Conv3d kernel = (tubelet_t, p, p), stride same → non-overlapping tubelets
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=(tubelet_t, patch_size, patch_size),
            stride=(tubelet_t, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: video [B, C, T, H, W]
        Returns:
            tokens [B, T', N, D]
        """
        x = self.proj(x)  # [B, D, T', H', W']
        B, D, Tp, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # [B, T', H', W', D]
        return x.reshape(B, Tp, Hp * Wp, D)


class FactorisedPosEmbed(nn.Module):
    """Separate spatial + temporal positional embeddings.

    pos[t, n] = space[n] + time[t]

    Benefits vs one big (T·N) table:
      • changing T only requires a longer time table
      • fewer parameters: O(T + N) instead of O(T·N)
    """

    def __init__(self, num_temporal: int, num_spatial: int, hidden_size: int):
        super().__init__()
        self.space = nn.Parameter(torch.zeros(1, 1, num_spatial, hidden_size))
        self.time = nn.Parameter(torch.zeros(1, num_temporal, 1, hidden_size))
        nn.init.trunc_normal_(self.space, std=0.02)
        nn.init.trunc_normal_(self.time, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add pos to tokens [B, T, N, D]."""
        T, N = x.shape[1], x.shape[2]
        return x + self.space[:, :, :N] + self.time[:, :T]


# ═══════════════════════════════════════════════════════════════════════
# 2. Efficient attention building blocks
# ═══════════════════════════════════════════════════════════════════════


class SpatialAttention(nn.Module):
    """Self-attention **within each frame** (space only).

    Tokens shaped [B, T, N, D] are reshaped to [B·T, N, D] so all frames
    share one batched matmul — no Python loop over T.
    Complexity: O(T · N² · D) instead of O((T·N)² · D).
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        h = self.norm(x).reshape(B * T, N, D)
        h = self.attn(h).reshape(B, T, N, D)
        return x + h


class TemporalAttention(nn.Module):
    """Self-attention **along time for each spatial location**.

    Reshape [B, T, N, D] → [B·N, T, D]: each patch trajectory attends
    over frames only.  Complexity: O(N · T² · D).
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        # [B, T, N, D] → [B, N, T, D] → [B·N, T, D]
        h = self.norm(x).permute(0, 2, 1, 3).reshape(B * N, T, D)
        h = self.attn(h).reshape(B, N, T, D).permute(0, 2, 1, 3)
        return x + h


class LocalTemporalAttention(nn.Module):
    """Temporal attention restricted to a local window of ±window frames.

    For long videos, full temporal attention is O(T²).  Local windows
    make it O(T · window) while still mixing nearby motion.

    Implementation: build an additive attention mask that sets scores
    outside [t−w, t+w] to −inf, then run standard attention on [B·N, T, D].
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.window = window
        self.norm = nn.LayerNorm(hidden_size)
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)

    def _mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Boolean mask [T, T]: True where attention is allowed."""
        idx = torch.arange(T, device=device)
        # |i − j| <= window
        return (idx[None, :] - idx[:, None]).abs() <= self.window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        h = self.norm(x).permute(0, 2, 1, 3).reshape(B * N, T, D)

        qkv = self.qkv(h).reshape(B * N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, BN, heads, T, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [BN, heads, T, T]
        # mask out frames outside the local window
        allow = self._mask(T, x.device)  # [T, T]
        attn = attn.masked_fill(~allow, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B * N, T, D)
        out = self.proj(out).reshape(B, N, T, D).permute(0, 2, 1, 3)
        return x + out


class DividedSpaceTimeBlock(nn.Module):
    """TimeSformer block: temporal attn → spatial attn → MLP.

    Order matters a little; paper default is time then space.
    Each sub-layer is residual + pre-norm style via the modules above.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temporal = TemporalAttention(hidden_size, num_heads, dropout)
        self.spatial = SpatialAttention(hidden_size, num_heads, dropout)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LocalSpaceTimeBlock(nn.Module):
    """Spatial full attention + local-window temporal attention + MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temporal = LocalTemporalAttention(hidden_size, num_heads, window, dropout)
        self.spatial = SpatialAttention(hidden_size, num_heads, dropout)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal(x)
        x = self.spatial(x)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class JointSpaceTimeBlock(nn.Module):
    """Full joint attention over flattened (T·N) tokens — expensive baseline."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        h = x.reshape(B, T * N, D)
        h = self.block(h)
        return h.reshape(B, T, N, D)


class SpaceOnlyBlock(nn.Module):
    """Spatial attention only (no temporal mixing inside the block)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spatial = SpatialAttention(hidden_size, num_heads, dropout)
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = x + self.mlp(self.norm_mlp(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
# 3. Full VideoViT
# ═══════════════════════════════════════════════════════════════════════


class VideoViT(nn.Module):
    """Video Vision Transformer with pluggable efficient attention modes.

    Args:
        attn_mode:
            ``space_only`` | ``divided`` | ``factorized`` |
            ``local_temporal`` | ``joint``
        tubelet_t:
            temporal size of each tubelet (≥1).  Larger → fewer tokens.
        temporal_window:
            only used by ``local_temporal`` (radius in frame units of T').
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_t: int = 2,
        in_channels: int = 3,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        num_temporal_layers: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 400,
        attn_mode: AttnMode = "divided",
        temporal_window: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert attn_mode in (
            "space_only",
            "divided",
            "factorized",
            "local_temporal",
            "joint",
        )
        self.attn_mode = attn_mode
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_t = tubelet_t
        self.hidden_size = hidden_size

        # ── tubelet / patch embedding ─────────────────────────────
        self.embed = TubeletEmbed(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_t=tubelet_t,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )
        self.num_temporal = self.embed.num_temporal  # T'
        self.num_spatial = self.embed.num_spatial  # N

        self.pos_embed = FactorisedPosEmbed(
            self.num_temporal, self.num_spatial, hidden_size
        )
        self.pos_drop = nn.Dropout(dropout)

        # optional CLS (used for pooling / factorized temporal encoder)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── backbone blocks depend on attn_mode ───────────────────
        if attn_mode == "divided":
            self.blocks = nn.ModuleList([
                DividedSpaceTimeBlock(hidden_size, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ])
        elif attn_mode == "local_temporal":
            self.blocks = nn.ModuleList([
                LocalSpaceTimeBlock(
                    hidden_size, num_heads, temporal_window, mlp_ratio, dropout
                )
                for _ in range(num_layers)
            ])
        elif attn_mode == "joint":
            self.blocks = nn.ModuleList([
                JointSpaceTimeBlock(hidden_size, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ])
        elif attn_mode in ("space_only", "factorized"):
            # spatial encoder shared; factorized adds a small temporal stack later
            self.blocks = nn.ModuleList([
                SpaceOnlyBlock(hidden_size, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(attn_mode)

        # ViViT-style factorized temporal encoder on per-frame descriptors
        self.temporal_blocks: nn.ModuleList | None
        if attn_mode == "factorized":
            self.temporal_blocks = nn.ModuleList([
                TransformerBlock(hidden_size, num_heads, mlp_ratio, dropout)
                for _ in range(num_temporal_layers)
            ])
            self.temporal_pos = nn.Parameter(
                torch.zeros(1, 1 + self.num_temporal, hidden_size)
            )
            nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        else:
            self.temporal_blocks = None
            self.temporal_pos = None

        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    # ── pooling helpers ───────────────────────────────────────────

    def _pool_space_time(self, x: torch.Tensor) -> torch.Tensor:
        """Global average pool over time & space → [B, D]."""
        return x.mean(dim=(1, 2))

    def _factorized_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Run temporal Transformer on per-frame mean-pooled tokens.

        x: [B, T, N, D] → frame feats [B, T, D] → + CLS → temporal blocks
        → CLS [B, D]
        Complexity O(T²) on a short sequence of length T (+1 CLS).
        """
        assert self.temporal_blocks is not None and self.temporal_pos is not None
        B, T, _, D = x.shape
        frame_tokens = x.mean(dim=2)  # [B, T, D]  spatial pool per frame

        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        seq = torch.cat([cls, frame_tokens], dim=1)  # [B, 1+T, D]
        seq = seq + self.temporal_pos[:, : 1 + T]

        for block in self.temporal_blocks:
            seq = block(seq)
        return seq[:, 0]  # temporal CLS

    # ── forward ───────────────────────────────────────────────────

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W]
        Returns:
            feature [B, D]
        """
        x = self.embed(video)  # [B, T', N, D]
        x = self.pos_drop(self.pos_embed(x))

        for block in self.blocks:
            x = block(x)

        if self.attn_mode == "factorized":
            feat = self._factorized_temporal(x)
        else:
            feat = self._pool_space_time(x)

        return self.norm(feat)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W]
        Returns:
            logits [B, num_classes]
        """
        return self.head(self.forward_features(video))


# ═══════════════════════════════════════════════════════════════════════
# 4. Complexity helper + demo
# ═══════════════════════════════════════════════════════════════════════


def approx_attn_flops(
    mode: AttnMode,
    T: int,
    N: int,
    D: int,
    layers: int,
    window: int = 2,
) -> int:
    """Rough attention FLOPs (matmul only) to compare modes.

    Counts dominant QKᵀ / AV costs as 2 · seq² · D per attention call.
    """

    def attn_cost(seq: int) -> int:
        return 2 * seq * seq * D

    if mode == "joint":
        per = attn_cost(T * N)
    elif mode == "space_only":
        per = T * attn_cost(N)
    elif mode == "divided":
        per = N * attn_cost(T) + T * attn_cost(N)
    elif mode == "factorized":
        # spatial layers + short temporal on T tokens (approx)
        per = T * attn_cost(N) + attn_cost(T + 1)
    elif mode == "local_temporal":
        # each of T positions attends to ~min(T, 2w+1) keys
        w = min(T, 2 * window + 1)
        temporal = N * (2 * T * w * D)
        spatial = T * attn_cost(N)
        per = temporal + spatial
    else:
        raise ValueError(mode)
    return per * layers


@torch.no_grad()
def benchmark_modes(
    img_size: int = 32,
    patch_size: int = 8,
    num_frames: int = 8,
    tubelet_t: int = 2,
    hidden_size: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    batch_size: int = 2,
    device: str | torch.device = "cpu",
) -> None:
    """Forward once per mode; print shape, params, latency, rough FLOPs."""
    modes: list[AttnMode] = [
        "space_only",
        "divided",
        "factorized",
        "local_temporal",
        "joint",
    ]
    video = torch.randn(batch_size, 3, num_frames, img_size, img_size, device=device)
    T_tokens = num_frames // tubelet_t
    N = (img_size // patch_size) ** 2

    logger.info(
        "Video shape %s | tokens T'=%d N=%d | device=%s",
        tuple(video.shape),
        T_tokens,
        N,
        device,
    )
    print(f"{'mode':<16} {'params':>10} {'ms':>10} {'attn FLOPs~':>14} {'out':>12}")
    print("-" * 68)

    for mode in modes:
        model = VideoViT(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_t=tubelet_t,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            num_temporal_layers=2,
            num_classes=10,
            attn_mode=mode,
            temporal_window=2,
        ).to(device)
        model.eval()

        # warmup
        _ = model(video)
        if device == "cuda" or (
            isinstance(device, torch.device) and device.type == "cuda"
        ):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        reps = 10
        for _ in range(reps):
            out = model(video)
        if device == "cuda" or (
            isinstance(device, torch.device) and device.type == "cuda"
        ):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / reps

        params = sum(p.numel() for p in model.parameters())
        flops = approx_attn_flops(mode, T_tokens, N, hidden_size, num_layers, window=2)
        assert out.shape == (batch_size, 10), out.shape
        print(
            f"{mode:<16} {params:>10} {ms:>10.2f} {flops:>14} {str(tuple(out.shape)):>12}"
        )


def smoke_test(device: str | torch.device = "cpu") -> None:
    """Assert every mode runs and returns correct logits shape."""
    B, C, T, H, W = 2, 3, 8, 32, 32
    video = torch.randn(B, C, T, H, W, device=device)
    for mode in ("space_only", "divided", "factorized", "local_temporal", "joint"):
        model = VideoViT(
            img_size=H,
            patch_size=8,
            num_frames=T,
            tubelet_t=2,
            hidden_size=64,
            num_heads=4,
            num_layers=2,
            num_temporal_layers=2,
            num_classes=5,
            attn_mode=mode,  # type: ignore[arg-type]
        ).to(device)
        out = model(video)
        assert out.shape == (B, 5), (mode, out.shape)
        logger.info("  mode=%-16s OK  out=%s", mode, tuple(out.shape))
    logger.info("smoke_test passed")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device=%s", device)
    smoke_test(device)
    print()
    benchmark_modes(device=device)
    print()
    logger.info(
        "Tip: prefer 'divided' or 'local_temporal' for real videos; "
        "use 'joint' only for tiny demos."
    )


if __name__ == "__main__":
    main()
