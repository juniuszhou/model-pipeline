"""A minimal Denoising Diffusion Probabilistic Model (DDPM) for image generation.

────────────────────────────────────────────────────────────────────────
  HOW DIFFUSION WORKS — a quick walkthrough
────────────────────────────────────────────────────────────────────────

  The idea is simple: take an image, gradually add noise until it becomes
  pure Gaussian noise (forward process), then teach a neural network to
  reverse that process (reverse process).

  Forward (noising) ─────►  x_0 ──► x_1 ──► … ──► x_T ≈ N(0,I)

  Reverse (denoising)  ◄────  x_0 ◄── x_1 ◄── … ◄── x_T
                                ^--- learned by ε_θ(x_t, t) ---^

  Training (one step):
    1. Sample a real image x_0 and a random timestep t
    2. Sample noise ε ~ N(0,I)
    3. Compute x_t = √(α̅_t)·x_0 + √(1-α̅_t)·ε   (closed-form noising)
    4. Predict ε_θ(x_t, t) ≈ ε
    5. Loss = MSE(ε_θ, ε)

  Sampling (generation):
    1. Start from pure noise x_T ~ N(0,I)
    2. For t = T, T−1, …, 1:
         a. Predict noise ε_θ(x_t, t)
         b. Compute x_{t-1} from x_t using the DDPM update formula
    3. Return x_0 — a generated image!

  References:
    - Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models", 2020
    - Nichol & Dhariwal. "Improved Denoising Diffusion Probabilistic Models", 2021
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)

# ── Schedule ──────────────────────────────────────────────────────────


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Improved DDPM).

    Smoother than the linear schedule used in the original DDPM paper.
    The idea: β_t should be small at first (images are fragile to noise)
    and larger later (when the signal is already mostly destroyed).
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(0.0001, 0.02)


# ── Time embedding (Transformer-style sinusoidal PE) ────────────────


def sinusoidal_time_embedding(
    t: Int[Tensor, "batch"], dim: int
) -> Float[Tensor, "batch seq dim"]:
    """Sinusoidal time embedding (same as Transformer positional encoding).

    Maps each scalar timestep t into a ``dim``-dimensional vector using
    sine/cosine functions of different frequencies.  This lets the network
    "know" which noise-level it's currently processing.
    """
    half = dim // 2
    freqs = torch.exp(-np.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


# ── Synthetic dataset ────────────────────────────────────────────────


def make_shape_dataset(
    num: int = 2000, size: int = 8
) -> Int[Tensor, "num 1 size size"]:
    """Generate synthetic ``size×size`` grayscale images with simple shapes.

    Three shape types, equally likely:
      * **circle** — filled disc at a random position with random radius
      * **cross**  — horizontal + vertical line through a random centre
      * **bar**    — horizontal bar of width 3 at a random row

    Pixel values are in [0, 1] (data scale), later normalised to [-1, 1].
    """
    imgs: Int[Tensor, "num 1 size size"] = torch.zeros(num, 1, size, size)
    rng = np.random.default_rng()
    for i in range(num):
        kind = rng.integers(0, 3)
        if kind == 0:  # circle
            cx, cy = rng.integers(1, size - 1, size=2)
            rsq = rng.uniform(0.5, 2.0) ** 2
            for x in range(size):
                for y in range(size):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= rsq:
                        imgs[i, 0, x, y] = 1.0
        elif kind == 1:  # cross
            cx, cy = rng.integers(0, size, size=2)
            imgs[i, 0, cx, :] = 1.0
            imgs[i, 0, :, cy] = 1.0
        else:  # bar
            row = rng.integers(0, size)
            col = rng.integers(0, size - 2)
            imgs[i, 0, row, col : col + 3] = 1.0
    return imgs


# ── Denoiser ──────────────────────────────────────────────────────────


class SimpleDenoiser(nn.Module):
    """A minimal ConvNet that predicts the noise added to a ``size×size`` image.

    Architecture (3 conv layers, 16 hidden channels):

        Input (1 ch)  ──► Conv2d(1→16) ──► SiLU ──► FiLM(time) ──►
            Conv2d(16→16) ──► SiLU ──► Conv2d(16→1) ──► noise (1 ch)

    Time is injected via *FiLM* (Feature-wise Linear Modulation):
    the sinusoidal time embedding is projected to a scale and a shift
    that multiply and add to the hidden activations.
    """

    def __init__(self, img_size: int = 8, time_dim: int = 16):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            # SiLU: x * sigmoid(x) no any parameters to train. we can use it as a activation function.
            nn.SiLU(),
            nn.Linear(time_dim, time_dim * 2),  # produces scale + shift
        )
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise ε that was added to the original image.

        Args:
            x: Noisy image at timestep t  [B, 1, H, W].
            t: Timestep indices  [B] (integers in 0..T-1).

        Returns:
            Predicted noise  [B, 1, H, W] (same shape as input).
        """
        te = sinusoidal_time_embedding(t, 16)  # [B, 16]
        scale, shift = self.time_mlp(te).chunk(2, dim=-1)  # each [B, 16]
        scale = scale[..., None, None]  # [B, 16, 1, 1]
        shift = shift[..., None, None]

        h = self.conv1(x)  # [B, 16, H, W]
        h = F.silu(h)
        h = h * (1.0 + scale) + shift  # FiLM

        h = self.conv2(h)
        h = F.silu(h)

        return self.conv3(h)  # [B, 1, H, W]


# ── The diffusion process ────────────────────────────────────────────


class Diffusion:
    """DDPM forward (noising) and reverse (denoising) processes.

    Usage::

        model = SimpleDenoiser()
        diff = Diffusion(timesteps=100)

        # Train
        losses = diff.train(model, dataset, epochs=50)

        # Generate
        samples = diff.sample(model, num=16)
    """

    def __init__(self, timesteps: int = 100):
        self.T = timesteps

        # ── Precompute all noise-schedule quantities ────────────────
        # These are used in both training and sampling.  Computing them
        # once in __init__ avoids recomputing them every step.

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas  # α_t = 1 - β_t
        alphas_cumprod = alphas.cumprod(dim=0)  # α̅_t = ∏_{s=1}^{t} α_s

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

        # Useful for the forward process (closed-form noising)
        self.sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    # ── Forward (noising) ───────────────────────────────────────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from the forward noising distribution q(x_t | x_0).

        Instead of iterating the Markov chain T times, DDPM uses the
        closed form (reparameterisation trick):

            x_t = √(α̅_t) · x_0  +  √(1 - α̅_t) · ε      ε ~ N(0, I)

        This is *the* key insight that makes training tractable — we can
        sample **any** timestep directly without simulating t steps.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_1mac * noise

    # ── Training ───────────────────────────────────────────────────

    def train(
        self,
        model: nn.Module,
        dataset: torch.Tensor,
        epochs: int = 100,
        lr: float = 3e-4,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        progress: bool = True,
    ) -> list[float]:
        """Train the denoiser to predict the noise added at each timestep.

        Training objective (the "simple" loss from the DDPM paper):

            L_simple = E_{t, x_0, ε} [ ‖ ε - ε_θ(x_t, t) ‖² ]

        where t ∼ Uniform(0, T-1), ε ∼ N(0,I), and x_t = q_sample(x_0, t, ε).

        Intuitively, the model learns to "undo" the noising process at
        **every** noise level simultaneously — from barely noisy (t ≈ 0)
        to almost pure noise (t ≈ T).
        """
        model.to(device)
        dataset = dataset.to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        losses: list[float] = []

        iterator = trange(epochs, desc="Diffusion") if progress else range(epochs)
        for _ in iterator:
            idx = torch.randperm(len(dataset), device=device)[:batch_size]
            x0 = dataset[idx]

            t = torch.randint(0, self.T, (batch_size,), device=device)
            noise = torch.randn_like(x0)
            xt = self.q_sample(x0, t, noise)
            pred = model(xt, t)
            loss = F.mse_loss(pred, noise)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            if progress:
                iterator.set_postfix(loss=loss.item())

        return losses

    # ── Sampling (generation) ──────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        num: int = 16,
        device: str | torch.device = "cpu",
        progress: bool = True,
    ) -> torch.Tensor:
        """Generate images by iteratively denoising pure Gaussian noise.

        The reverse (denoising) process starts from x_T ~ N(0,I) and
        applies the learned denoising transition for t = T-1, …, 0:

            x_{t-1} = 1/√(α_t) · ( x_t - β_t / √(1 - α̅_t) · ε_θ(x_t, t) )
                    + σ_t · z

        where:
            z ∼ N(0,I) for t > 0, z = 0 for t = 0 (the final step)
            σ_t² = β_t     (variance of the reverse process)

        This is called **ancestral sampling** (a.k.a. the DDPM sampler)
        because each step samples from p_θ(x_{t-1} | x_t).
        """
        model.eval()
        model.to(device)

        xt = torch.randn(num, 1, 8, 8, device=device)

        iterator = (
            tqdm(reversed(range(self.T)), desc="Sample", total=self.T)
            if progress
            else reversed(range(self.T))
        )
        for t_int in iterator:
            t = torch.full((num,), t_int, device=device, dtype=torch.long)

            pred_noise = model(xt, t)

            α = self.alphas[t_int]
            α̅ = self.alphas_cumprod[t_int]
            β = self.betas[t_int]

            # DDPM update
            x_prev = (xt - β / (1 - α̅).sqrt() * pred_noise) / α.sqrt()

            # Add Langevin noise (except at the very last step)
            if t_int > 0:
                x_prev = x_prev + β.sqrt() * torch.randn_like(xt)

            xt = x_prev

        # Clamp to valid pixel range [-1, 1]
        return xt.clamp(-1.0, 1.0)


# ── Visualisation ─────────────────────────────────────────────────────


def save_results(
    generated: torch.Tensor,
    losses: list[float],
    img_size: int = 8,
    path: str = "diffusion_result.png",
) -> None:
    """Saves a grid of generated images alongside the training loss curve."""
    num = generated.shape[0]
    cols = int(np.ceil(np.sqrt(num)))
    rows = int(np.ceil(num / cols))

    fig, (ax_grid, ax_loss) = plt.subplots(1, 2, figsize=(10, 5))

    # Image grid
    grid = np.zeros((rows * (img_size + 1) - 1, cols * (img_size + 1) - 1))
    for i in range(num):
        r = i // cols
        c = i % cols
        yr = r * (img_size + 1)
        xr = c * (img_size + 1)
        grid[yr : yr + img_size, xr : xr + img_size] = (
            generated[i, 0].cpu().numpy() * 0.5 + 0.5  # map [-1,1] → [0,1]
        )
    ax_grid.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax_grid.set_title("Generated images")
    ax_grid.axis("off")

    # Loss curve
    ax_loss.plot(losses)
    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("MSE")
    ax_loss.set_yscale("log")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info("Result saved to %s", path)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # ── Dataset ────────────────────────────────────────────────────
    size = 8
    dataset = make_shape_dataset(num=2000, size=size)
    # Normalise pixel values from [0, 1] to [-1, 1] (common practice
    # for diffusion models — matches the Gaussian noise scale).
    dataset = dataset * 2.0 - 1.0
    logger.info(
        "Dataset shape: %s  (range [%.1f, %.1f])",
        dataset.shape,
        dataset.min(),
        dataset.max(),
    )

    # ── Model ──────────────────────────────────────────────────────
    model = SimpleDenoiser(img_size=size)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Denoiser parameters: %d", n_params)

    # ── Diffusion ──────────────────────────────────────────────────
    diffusion = Diffusion(timesteps=100)

    logger.info("Starting training …")
    losses = diffusion.train(model, dataset, epochs=80, batch_size=128, device=device)
    logger.info("Training done.  Final loss: %.6f", losses[-1])

    # ── Sample ─────────────────────────────────────────────────────
    logger.info("Generating images …")
    generated = diffusion.sample(model, num=25, device=device)

    # ── Save ───────────────────────────────────────────────────────
    save_results(generated, losses, img_size=size, path="diffusion_result.png")
    logger.info("All done!  Check diffusion_result.png for the output.")


if __name__ == "__main__":
    main()
