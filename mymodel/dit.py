"""
Diffusion Transformer (DiT) — text-conditioned image generation demo.

────────────────────────────────────────────────────────────────────────
  WHAT IS DiT?
────────────────────────────────────────────────────────────────────────

  Standard DDPM (see dmodel.py) uses a *U-Net* to predict noise.
  DiT replaces that U-Net with a *Vision Transformer* on image patches.
  Same diffusion math, different neural backbone.

  This demo adds **text conditioning** so you can generate images from
  prompts like "a circle" or "draw a cross":

      text prompt  ──►  TextEmbedder  ──►  y  [B, D]
      timestep t   ──►  TimestepEmbedder ──►  t_emb  [B, D]
                              │
                         c = t_emb + y     (condition vector)
                              │
      Image x_t  ──►  Patchify  ──►  tokens
                         │
                         ▼
              ┌─────────────────────┐
              │  DiT Block  × L     │  ← each block uses c via adaLN
              │  (Attention + MLP)  │
              └─────────────────────┘
                         │
                         ▼
                 Unpatchify  ──►  predicted noise ε̂

  Classifier-free guidance (CFG):
    At training time we randomly drop the text (use a null embedding).
    At sampling time we run both conditional and unconditional predictions
    and mix them:

        ε = ε_uncond + guidance_scale · (ε_cond − ε_uncond)

    Higher guidance_scale → stronger adherence to the text prompt.

  References:
    - Peebles & Xie. "Scalable Diffusion Models with Transformers", 2023
    - Ho & Salimans. "Classifier-Free Diffusion Guidance", 2022
    - Ho et al. "Denoising Diffusion Probabilistic Models", 2020
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. Noise schedule
# ═══════════════════════════════════════════════════════════════════════


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from Improved DDPM.

    Returns a 1-D tensor ``betas`` of length ``timesteps``.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0.0001, 0.02)


# ═══════════════════════════════════════════════════════════════════════
# 2. Timestep embedding
# ═══════════════════════════════════════════════════════════════════════


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map integer timesteps → dense vectors with sine / cosine.

    Args:
        t:   integer timesteps, shape [B]
        dim: embedding dimension (must be even)

    Returns:
        embeddings of shape [B, dim]
    """
    half = dim // 2
    freqs: Float[Tensor, "half"] = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    # t is a tensor of shape [B], go to [B, 1]
    # freqs is a tensor of shape [half], go to [1, half]
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]

    # concat the sin and cos of the arguments
    return torch.cat([args.sin(), args.cos()], dim=-1)  # [B, dim]


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding → 2-layer MLP → vector of size hidden_size."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # result is [B, hidden_size]
        return self.mlp(sinusoidal_embedding(t, self.freq_dim))


# ═══════════════════════════════════════════════════════════════════════
# 3. Text conditioning  (prompt → embedding y)
# ═══════════════════════════════════════════════════════════════════════

# Special token ids (must match order in VOCAB)
PAD_ID = 0
UNK_ID = 1
NULL_ID = 2  # used for classifier-free guidance ("no text")

# Tiny vocabulary for this demo.  Real systems use CLIP / T5 with ~50k tokens.
# We intentionally keep it small so the whole text encoder trains quickly.
VOCAB: list[str] = [
    "[PAD]",
    "[UNK]",
    "[NULL]",
    "a",
    "an",
    "the",
    "draw",
    "generate",
    "make",
    "show",
    "picture",
    "image",
    "shape",
    "of",
    "circle",
    "round",
    "disk",
    "dot",
    "filled",
    "cross",
    "plus",
    "x",
    "intersecting",
    "lines",
    "bar",
    "line",
    "horizontal",
    "strip",
    "thick",
]
WORD2ID: dict[str, int] = {w: i for i, w in enumerate(VOCAB)}
MAX_TEXT_LEN = 8

# Human-readable class names and training captions for each shape.
CLASS_NAMES: dict[int, str] = {0: "circle", 1: "cross", 2: "bar"}
CLASS_CAPTIONS: dict[int, list[str]] = {
    0: [
        "circle",
        "a circle",
        "a round shape",
        "filled circle",
        "draw a circle",
        "a disk",
        "generate a circle",
        "a round disk",
    ],
    1: [
        "cross",
        "a cross",
        "a plus",
        "draw a cross",
        "intersecting lines",
        "generate a cross",
        "an x shape",
        "a plus shape",
    ],
    2: [
        "bar",
        "a bar",
        "horizontal bar",
        "a thick bar",
        "draw a bar",
        "a horizontal line",
        "generate a bar",
        "a strip",
    ],
}


def tokenize(text: str, max_len: int = MAX_TEXT_LEN) -> list[int]:
    """Very simple whitespace / punctuation tokenizer.

    - lowercases the string
    - splits on non-alphanumeric characters
    - maps known words to ids, unknown → [UNK]
    - pads / truncates to ``max_len``
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    ids = [WORD2ID.get(tok, UNK_ID) for tok in tokens][:max_len]
    if not ids:
        ids = [UNK_ID]
    # pad to fixed length so we can batch easily
    ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids


def tokenize_batch(
    texts: list[str],
    max_len: int = MAX_TEXT_LEN,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Tokenize a list of prompts → LongTensor [B, max_len]."""
    return torch.tensor(
        [tokenize(t, max_len) for t in texts],
        dtype=torch.long,
        device=device,
    )


def null_tokens(
    batch_size: int,
    max_len: int = MAX_TEXT_LEN,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Tokens for the empty / dropped condition (classifier-free guidance)."""
    ids = [NULL_ID] + [PAD_ID] * (max_len - 1)
    return torch.tensor([ids] * batch_size, dtype=torch.long, device=device)


class TextEmbedder(nn.Module):
    """Tiny text encoder: word embeddings → mean pool → MLP.

    In production systems this would be CLIP or T5.  Here we train a
    small embedding table *jointly* with the diffusion model so the
    demo has no external dependencies.

    Input:  token ids  [B, L]
    Output: text vector y  [B, hidden_size]
    """

    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        hidden_size: int = 64,
        word_dim: int = 32,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, word_dim, padding_idx=PAD_ID)
        self.mlp = nn.Sequential(
            nn.Linear(word_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] long tensor
        Returns:
            y: [B, hidden_size]
        """
        emb = self.embed(token_ids)  # [B, L, word_dim]
        # Mask out padding so it does not dilute the mean
        mask = (token_ids != PAD_ID).float().unsqueeze(-1)  # [B, L, 1]
        summed = (emb * mask).sum(dim=1)  # [B, word_dim]
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts  # mean over non-pad tokens
        return self.mlp(pooled)


# ═══════════════════════════════════════════════════════════════════════
# 4. Patchify / Unpatchify
# ═══════════════════════════════════════════════════════════════════════


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split an image into non-overlapping patches and flatten each patch.

    Args:
        x:          [B, C, H, W]
        patch_size: side length of one square patch

    Returns:
        patches [B, N, P] with N=(H/p)*(W/p), P=C*p*p
    """
    B, C, H, W = x.shape
    p = patch_size
    assert H % p == 0 and W % p == 0, (
        f"H={H}, W={W} must be divisible by patch_size={p}"
    )

    x = x.reshape(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, nH, nW, C, p, p]
    return x.reshape(B, (H // p) * (W // p), C * p * p)


def unpatchify(
    patches: torch.Tensor, patch_size: int, img_size: int, channels: int = 1
) -> torch.Tensor:
    """Inverse of patchify: sequence of patches → image [B, C, H, W]."""
    B, N, _ = patches.shape
    p = patch_size
    n = img_size // p
    assert N == n * n

    x = patches.reshape(B, n, n, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, channels, img_size, img_size)


# ═══════════════════════════════════════════════════════════════════════
# 5. Core DiT layers
# ═══════════════════════════════════════════════════════════════════════


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention (same as ViT / GPT)."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = self.head_dim**-0.5
        # last two dimensions of attention matrix are (N, N) it is hidden size or hidden size / num_heads for multi head
        attn: Float[Tensor, "B heads N N"] = (q @ k.transpose(-2, -1)) * scale
        attn: Float[Tensor, "B heads N N"] = attn.softmax(dim=-1)
        out: Float[Tensor, "B heads N head_dim"] = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise MLP: Linear(D→4D) → GELU → Linear(4D→D)."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN core: y = x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """One DiT block with adaptive LayerNorm (adaLN-Zero).

    The 6 modulation parameters come from the *combined* condition
    c = t_emb + text_emb, so both time and text steer every layer.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Predict 6 modulation vectors from condition c = t_emb + text_emb.
        # Each is shape [B, D].  They come in two groups:
        #
        #   ── Attention branch (msa = multi-head self-attention) ──
        #   shift_msa : additive bias after LayerNorm   →  h = h + shift
        #   scale_msa : multiplicative gain after LN    →  h = h * (1 + scale)
        #               Together, shift+scale = adaLN / FiLM: they let time & text
        #               reshape the *features that enter attention*.
        #   gate_msa  : residual gate on the attention output
        #               →  x = x + gate_msa * Attention(h)
        #               Starts at 0 (zero-init), so the block is initially identity.
        #               Training gradually opens the gate as attention becomes useful.
        #
        #   ── MLP / feed-forward branch ──
        #   shift_mlp : same role as shift_msa, but for the MLP input
        #   scale_mlp : same role as scale_msa, but for the MLP input
        #   gate_mlp  : residual gate on the MLP output
        #               →  x = x + gate_mlp * MLP(h)
        #
        # Formula recap:
        #   h_attn = LayerNorm(x) * (1 + scale_msa) + shift_msa
        #   x      = x + gate_msa * Attention(h_attn)
        #   h_mlp  = LayerNorm(x) * (1 + scale_mlp) + shift_mlp
        #   x      = x + gate_mlp * MLP(h_mlp)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(h)
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """adaLN → linear projection back to patch pixels."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # back to patch pixels
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ═══════════════════════════════════════════════════════════════════════
# 6. Full DiT model  (with text condition)
# ═══════════════════════════════════════════════════════════════════════


class DiT(nn.Module):
    """Text-conditioned Diffusion Transformer.

    Predicts noise ε given noisy image x_t, timestep t, and text tokens.

        c = TimestepEmbedder(t) + TextEmbedder(text)
        ε̂ = DiT_blocks(patchify(x_t); c)
    """

    def __init__(
        self,
        img_size: int = 8,
        patch_size: int = 2,
        in_channels: int = 1,
        hidden_size: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        vocab_size: int = len(VOCAB),
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # in channels is equal to out channels
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, hidden_size)
        self.register_buffer(
            "pos_embed",
            self._build_2d_sincos_pos_embed(hidden_size, img_size // patch_size),
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_embedder = TextEmbedder(
            vocab_size=vocab_size, hidden_size=hidden_size
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                if module.weight.std().item() == 0.0:
                    return
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    @staticmethod
    def _build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2-D sincos"
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=0)
        grid = grid.reshape(2, 1, grid_size, grid_size)
        emb_h = DiT._1d_sincos(embed_dim // 2, grid[0].reshape(-1))
        emb_w = DiT._1d_sincos(embed_dim // 2, grid[1].reshape(-1))
        pos = torch.cat([emb_h, emb_w], dim=1)
        return pos.unsqueeze(0)

    @staticmethod
    def _1d_sincos(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
        out = pos.unsqueeze(1) * omega.unsqueeze(0)
        return torch.cat([out.sin(), out.cos()], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise ε given x_t, timestep t, and text token ids.

        Args:
            x:        noisy images   [B, C, H, W]
            t:        timesteps      [B]
            text_ids: token ids      [B, L]

        Returns:
            predicted noise          [B, C, H, W]
        """
        tokens: Float[Tensor, "B N C_P_P"] = patchify(x, self.patch_size)
        tokens = self.patch_embed(tokens) + self.pos_embed

        # Combine time + text into one condition vector for adaLN
        c: Float[Tensor, "B dim"] = self.t_embedder(t) + self.text_embedder(
            text_ids
        )  # [B, D]

        for block in self.blocks:
            tokens = block(tokens, c)

        tokens = self.final_layer(tokens, c)
        predicted_noise = unpatchify(
            tokens, self.patch_size, self.img_size, self.out_channels
        )
        return predicted_noise


# ═══════════════════════════════════════════════════════════════════════
# 7. Diffusion process  (train + text-conditioned generate)
# ═══════════════════════════════════════════════════════════════════════


class Diffusion:
    """DDPM with text conditioning and classifier-free guidance."""

    def __init__(self, timesteps: int = 100):
        self.T = timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    def _to_device(self, device: str | torch.device) -> None:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
            device
        )

    # get a noisy image from the original image, noise and timestep
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Closed-form noising: x_t = √(α̅_t)·x_0 + √(1-α̅_t)·ε."""
        if noise is None:
            noise = torch.randn_like(x0)

        # get two parameters from cache
        a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # compute image add noise
        return a * x0 + b * noise

    def train(
        self,
        model: DiT,
        images: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
        lr: float = 3e-4,
        batch_size: int = 64,
        cfg_drop_prob: float = 0.1,
        device: str | torch.device = "cpu",
        progress: bool = True,
    ) -> list[float]:
        """Train text-conditioned DiT.

        For each sample we pick a random caption for its class label,
        tokenize it, and (with probability ``cfg_drop_prob``) replace
        it with the null token so the model also learns unconditional
        denoising — required for classifier-free guidance at sample time.
        """
        model.to(device)
        self._to_device(device)
        images = images.to(device)
        labels = labels.to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        losses: list[float] = []
        rng = np.random.default_rng(0)

        # here we don't iterate over the timesteps, we iterate over the epochs
        # time parameter will be sampled from the random function.
        iterator = trange(epochs, desc="DiT train") if progress else range(epochs)
        for _ in iterator:
            #  randomly select a batch of images and labels
            idx = torch.randperm(len(images), device=device)[:batch_size]
            x0 = images[idx]
            y = labels[idx]

            # Build a text batch: random caption per class, then maybe drop
            texts: list[str] = []
            #  from the label to get the description of the class such as "a circle", "a cross", "a bar"
            for label in y.tolist():
                caps = CLASS_CAPTIONS[int(label)]
                texts.append(caps[int(rng.integers(0, len(caps)))])
            text_ids = tokenize_batch(texts, device=device)

            # Classifier-free guidance training: randomly drop text
            drop = torch.rand(batch_size, device=device) < cfg_drop_prob
            if drop.any():
                text_ids[drop] = null_tokens(int(drop.sum().item()), device=device)

            t = torch.randint(0, self.T, (batch_size,), device=device)
            noise = torch.randn_like(x0)
            xt = self.q_sample(x0, t, noise)

            # model input xt is the original image + noise
            pred = model(xt, t, text_ids)
            # compare noise and predicted noise
            loss = F.mse_loss(pred, noise)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            if progress:
                iterator.set_postfix(loss=f"{loss.item():.4f}")

        return losses

    def _predict_noise(
        self,
        model: DiT,
        xt: torch.Tensor,
        t: torch.Tensor,
        text_ids: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Noise prediction with optional classifier-free guidance.

        ε = ε_uncond + s · (ε_cond − ε_uncond)

        When guidance_scale == 1.0 this reduces to plain conditional prediction.
        """
        if guidance_scale == 1.0:
            return model(xt, t, text_ids)

        # Run conditional and unconditional in one batched forward for speed
        b = xt.shape[0]

        # double the input image and noise
        xt_2 = torch.cat([xt, xt], dim=0)
        t_2 = torch.cat([t, t], dim=0)
        null = null_tokens(b, device=xt.device)

        # one input with text, one input without text
        text_2 = torch.cat([text_ids, null], dim=0)

        # model input xt_2 is the original image + noise
        # pred_2 is the predicted noise
        pred_2 = model(xt_2, t_2, text_2)

        pred_cond, pred_uncond = pred_2.chunk(2, dim=0)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    @torch.no_grad()
    def sample(
        self,
        model: DiT,
        text_ids: torch.Tensor,
        img_size: int = 8,
        channels: int = 1,
        guidance_scale: float = 3.0,
        device: str | torch.device = "cpu",
        progress: bool = True,
    ) -> torch.Tensor:
        """Denoise from pure noise, conditioned on tokenized text.

        Args:
            text_ids: [B, L] token ids (one prompt per image)
            guidance_scale: CFG strength (1 = no guidance, 3–7 typical)
        """
        model.eval()
        model.to(device)
        self._to_device(device)
        text_ids = text_ids.to(device)
        num = text_ids.shape[0]

        xt = torch.randn(num, channels, img_size, img_size, device=device)

        steps = (
            tqdm(reversed(range(self.T)), desc="Sample", total=self.T)
            if progress
            else reversed(range(self.T))
        )
        # when we generate the image, the timesteps should be reversed.
        # Because we want to start from the last timestep and go to the first timestep.
        # and last image should be a pure noise image.
        for t_int in steps:
            t = torch.full((num,), t_int, device=device, dtype=torch.long)
            pred_noise = self._predict_noise(model, xt, t, text_ids, guidance_scale)

            # parameters at the timestep t_int
            alpha = self.alphas[t_int]
            alpha_bar = self.alphas_cumprod[t_int]
            beta = self.betas[t_int]

            # formula: x_t = (x_t - beta / (1 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
            # use current more noisy image to get the previous image via subtracting the predicted noise
            x_prev = (xt - beta / (1 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
            if t_int > 0:
                x_prev = x_prev + beta.sqrt() * torch.randn_like(xt)
            xt = x_prev

        return xt.clamp(-1.0, 1.0)

    @torch.no_grad()
    def generate(
        self,
        model: DiT,
        prompts: str | list[str],
        num_per_prompt: int = 1,
        img_size: int = 8,
        channels: int = 1,
        guidance_scale: float = 3.0,
        device: str | torch.device = "cpu",
        progress: bool = True,
    ) -> tuple[torch.Tensor, list[str]]:
        """Generate images from natural-language prompts.

        This is the main user-facing API for text-to-image.

        Args:
            prompts: one string, or a list of strings
            num_per_prompt: how many images to sample per prompt
            guidance_scale: classifier-free guidance strength

        Returns:
            images:  [B, C, H, W] in [-1, 1]
            labels:  length-B list of the prompt used for each image

        Example::

            imgs, used = diffusion.generate(model, ["a circle", "draw a cross"])
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        expanded: list[str] = []
        for p in prompts:
            expanded.extend([p] * num_per_prompt)

        text_ids = tokenize_batch(expanded, device=device)
        images = self.sample(
            model,
            text_ids,
            img_size=img_size,
            channels=channels,
            guidance_scale=guidance_scale,
            device=device,
            progress=progress,
        )
        return images, expanded


# ═══════════════════════════════════════════════════════════════════════
# 8. Synthetic (image, label) dataset
# ═══════════════════════════════════════════════════════════════════════


def make_shape_dataset(
    num: int = 2000, size: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate greyscale shapes with class labels.

    Classes: 0=circle, 1=cross, 2=bar.  Pixels in [0, 1].

    Returns:
        images: [num, 1, size, size]
        labels: [num] long
    """
    imgs = torch.zeros(num, 1, size, size)
    labels = torch.zeros(num, dtype=torch.long)
    rng = np.random.default_rng(42)
    for i in range(num):
        kind = int(rng.integers(0, 3))
        labels[i] = kind
        if kind == 0:  # filled circle
            cx, cy = rng.integers(1, size - 1, size=2)
            rsq = float(rng.uniform(0.5, 2.0)) ** 2
            for x in range(size):
                for y in range(size):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= rsq:
                        imgs[i, 0, x, y] = 1.0
        elif kind == 1:  # cross
            cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
            imgs[i, 0, cx, :] = 1.0
            imgs[i, 0, :, cy] = 1.0
        else:  # horizontal bar
            row = int(rng.integers(0, size))
            col = int(rng.integers(0, max(1, size - 2)))
            imgs[i, 0, row, col : col + 3] = 1.0
    return imgs, labels


# ═══════════════════════════════════════════════════════════════════════
# 9. Visualisation
# ═══════════════════════════════════════════════════════════════════════


def save_results(
    generated: torch.Tensor,
    losses: list[float],
    prompts: list[str] | None = None,
    img_size: int = 8,
    path: str = "dit_result.png",
) -> None:
    """Save generated images (with optional prompt titles) + loss curve."""
    num = generated.shape[0]
    cols = int(math.ceil(math.sqrt(num)))
    rows = int(math.ceil(num / cols))

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
    ax_grid = fig.add_subplot(gs[0])
    ax_loss = fig.add_subplot(gs[1])

    # Draw each image in its own small axes for prompt labels
    ax_grid.set_axis_off()
    inner = gs[0].subgridspec(rows, cols, wspace=0.15, hspace=0.45)
    for i in range(num):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(inner[r, c])
        img = generated[i, 0].cpu().numpy() * 0.5 + 0.5
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if prompts is not None and i < len(prompts):
            ax.set_title(prompts[i], fontsize=7)

    ax_loss.plot(losses)
    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_yscale("log")

    fig.suptitle("DiT text-to-image", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info("Result saved to %s", path)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# 10. Demo
# ═══════════════════════════════════════════════════════════════════════


def smoke_test(device: str | torch.device = "cpu") -> None:
    """Quick forward-pass check of every component (no training)."""
    logger.info("── Smoke test ──────────────────────────────────────")
    B, C, H, p = 4, 1, 8, 2
    # height and width are the same
    x = torch.randn(B, C, H, H, device=device)
    # timestep is a random integer between 0 and 100
    t = torch.randint(0, 100, (B,), device=device)

    patches: Int[Tensor, "B C H H"] = patchify(x, p)
    assert torch.allclose(x, unpatchify(patches, p, H, C))
    logger.info("  patchify / unpatchify  OK")

    # Tokenizer + text embedder
    ids = tokenize_batch(
        ["a circle", "draw a cross", "bar", "unknown xyz"], device=device
    )
    assert ids.shape == (B, MAX_TEXT_LEN)
    te = TextEmbedder(hidden_size=64).to(device)
    y = te(ids)
    assert y.shape == (B, 64)
    logger.info("  TextEmbedder           OK  %s", tuple(y.shape))

    te_t = TimestepEmbedder(hidden_size=64).to(device)
    assert te_t(t).shape == (B, 64)
    logger.info("  TimestepEmbedder       OK")

    model = DiT(img_size=H, patch_size=p, hidden_size=64, depth=2, num_heads=4).to(
        device
    )

    # input, time and ids
    out = model(x, t, ids)
    assert out.shape == x.shape
    n_params = sum(p_.numel() for p_ in model.parameters())
    logger.info("  DiT + text forward     OK  params=%d", n_params)
    logger.info("── Smoke test passed ───────────────────────────────")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    smoke_test(device)

    img_size = 8
    patch_size = 2
    model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=1,
        hidden_size=64,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("DiT params: %d", n_params)

    # Dataset: images in [-1, 1] + integer class labels

    # images are in [0, 1]
    images, labels = make_shape_dataset(num=2000, size=img_size)
    # convert to [-1, 1]
    images = images * 2.0 - 1.0

    logger.info(
        "Dataset: images=%s labels=%s  (0=circle, 1=cross, 2=bar)",
        images.shape,
        labels.shape,
    )

    diffusion = Diffusion(timesteps=100)
    logger.info("Training with text captions + CFG dropout …")
    losses = diffusion.train(
        model,
        images,
        labels,
        epochs=120,  # a bit more epochs so text conditioning learns
        batch_size=128,
        lr=3e-4,
        cfg_drop_prob=0.1,
        device=device,
    )
    logger.info("Training done.  Final loss: %.6f", losses[-1])

    # ── Text-to-image generation ──────────────────────────────────
    prompts = [
        "a circle",
        "draw a circle",
        "a cross",
        "draw a cross",
        "a bar",
        "horizontal bar",
        "generate a circle",
        "a plus",
        "a thick bar",
    ]
    logger.info("Generating from prompts: %s", prompts)
    generated, used = diffusion.generate(
        model,
        prompts,
        num_per_prompt=1,
        img_size=img_size,
        guidance_scale=3.0,
        device=device,
    )

    save_results(
        generated,
        losses,
        prompts=used,
        img_size=img_size,
        path="dit_result.png",
    )
    logger.info("All done!  Open dit_result.png — each tile is labelled by its prompt.")
    logger.info(
        "Try your own prompts, e.g.\n"
        "  imgs, _ = diffusion.generate(model, ['a circle', 'draw a cross'], device=device)"
    )


if __name__ == "__main__":
    main()
