"""Train U-Net on a synthetic denoising task."""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mymodel.u_net import UNet


class SyntheticDenoiseDataset(Dataset):
    """
    Generate synthetic (noisy, clean) image pairs for denoising.

    Clean images are random smooth blobs (simulating natural image structure).
    Noisy inputs = clean + Gaussian noise.
    """

    def __init__(self, num_samples, img_size=64, in_channels=3, noise_std=0.3):
        self.num_samples = num_samples
        self.img_size = img_size
        self.in_channels = in_channels
        self.noise_std = noise_std

    def __len__(self):
        return self.num_samples

    def _make_clean(self, n):
        """Generate n clean images: random low-frequency blobs in [0,1]."""
        grid = torch.linspace(-1, 1, self.img_size)
        y, x = torch.meshgrid(grid, grid, indexing="ij")
        clean = torch.zeros(n, self.in_channels, self.img_size, self.img_size)
        for i in range(n):
            for c in range(self.in_channels):
                cx, cy = torch.rand(2) * 2 - 1
                sx, sy = 0.2 + torch.rand(2) * 0.6
                blob = torch.exp(-(((x - cx) / sx) ** 2) - ((y - cy) / sy) ** 2)
                blob = blob * (0.5 + torch.rand(1) * 0.5)
                clean[i, c] = blob
        return clean

    def __getitem__(self, idx):
        n = 1
        clean = self._make_clean(n).squeeze(0)
        noise = torch.randn_like(clean) * self.noise_std
        noisy = (clean + noise).clamp(0, 1)
        return noisy, clean


def main():
    parser = argparse.ArgumentParser(description="Train U-Net on image denoising")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--train_samples", type=int, default=2000)
    parser.add_argument("--val_samples", type=int, default=200)
    parser.add_argument("--noise_std", type=float, default=0.3)
    parser.add_argument("--output_dir", type=str, default="./unet_model")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Data
    train_ds = SyntheticDenoiseDataset(
        args.train_samples, args.img_size, args.in_channels, args.noise_std
    )
    val_ds = SyntheticDenoiseDataset(
        args.val_samples, args.img_size, args.in_channels, args.noise_std
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Model, loss, optimizer
    model = UNet(args.in_channels, args.out_channels, args.hidden_channels).to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False
        )
        for noisy, clean in pbar:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            loss = loss_fn(pred, clean)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                val_loss += loss_fn(pred, clean).item()
        val_loss /= len(val_loader)

        scheduler.step()

        print(
            f"Epoch {epoch:2d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            print(f"  → saved best model (val_loss={val_loss:.4f})")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final.pt"))
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")

    # Visualise a few validation results
    model.eval()
    noisy, clean = next(iter(val_loader))
    noisy, clean = noisy[:4].to(device), clean[:4].to(device)
    with torch.no_grad():
        pred = model(noisy)

    def to_img(t):
        return t.cpu().clamp(0, 1)

    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    for i in range(4):
        axes[i, 0].imshow(to_img(noisy[i]).permute(1, 2, 0))
        axes[i, 0].set_title("Noisy Input")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(to_img(pred[i]).permute(1, 2, 0))
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(to_img(clean[i]).permute(1, 2, 0))
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")
    fig.suptitle("Denoising Results")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "val_samples.png"), dpi=150)
    print(f"Saved validation samples to {args.output_dir}/val_samples.png")


if __name__ == "__main__":
    main()
