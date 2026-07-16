"""U-Net model for image-to-image tasks."""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling: MaxPool -> DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNet(nn.Module):
    """
    U-Net: encoder-bottleneck-decoder with explicit skip connections.

    Architecture:

         input
           │
        ┌──inc──────────────┬──────────────────────────────┐
        │                   │                              │
        down1───────────────┬─────────────────────────┐    │
        │                   │                         │    │
        down2───────────────┬────────────────────┐    │    │
        │                   │                    │    │    │
        down3───────────────┬───────────────┐    │    │    │
        │                   │               │    │    │    │
      bottleneck────────────┘               │    │    │    │
        │                            ┌──────┘    │    │    │
        │                            │ up_conv1  │    │    │
        │                            │ (+skip4)  │    │    │
        │                            └──────┐    │    │    │
        │                                   │ up_conv2   │
        │                                   │ (+skip3)   │
        │                                   └──────┐    │
        │                                          │ up_conv3
        │                                          │ (+skip2)
        │                                          └──────┐
        │                                                 │ up_conv4
        │                                                 │ (+skip1)
        │                                                 └──────┐
        │                                                        │
        └────────────────────────────────────────────────────────┤
                                                                 │
                                                              out_conv
                                                                 │
                                                               output

    每个 decoder 层先上采样，再与对应的 encoder 特征在通道维拼接(cat)，
    即显式的 skip connection，让梯度可直接回传到浅层，缓解深层的梯度消失。

    Shape flow (hidden_channels=32, input 128x128):
      input:  [B, 3, 128, 128]
      inc:    [B, 32, 128, 128]   → skip1
      down1:  [B, 64,  64,  64]   → skip2
      down2:  [B, 128, 32,  32]   → skip3
      down3:  [B, 256, 16,  16]   → skip4
      bottle: [B, 512,  8,   8]
      up_1:  cat(skip4, ↑bottle)  → [B, 256, 16, 16]
      up_2:  cat(skip3, ↑up_1)    → [B, 128, 32, 32]
      up_3:  cat(skip2, ↑up_2)    → [B, 64,  64, 64]
      up_4:  cat(skip1, ↑up_3)    → [B, 32, 128, 128]
      output: [B, out_c, 128, 128]
    """

    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Encoder (contracting path)
        self.inc = DoubleConv(in_channels, hidden_channels)
        self.down1 = Down(hidden_channels, hidden_channels * 2)
        self.down2 = Down(hidden_channels * 2, hidden_channels * 4)
        self.down3 = Down(hidden_channels * 4, hidden_channels * 8)

        # Bottleneck
        self.bottleneck = Down(hidden_channels * 8, hidden_channels * 16)

        # Decoder upsampling + DoubleConv (skip connection is done explicitly in forward)
        self.up_conv1 = DoubleConv(
            hidden_channels * 16 + hidden_channels * 8, hidden_channels * 8
        )
        self.up_conv2 = DoubleConv(
            hidden_channels * 8 + hidden_channels * 4, hidden_channels * 4
        )
        self.up_conv3 = DoubleConv(
            hidden_channels * 4 + hidden_channels * 2, hidden_channels * 2
        )
        self.up_conv4 = DoubleConv(
            hidden_channels * 2 + hidden_channels, hidden_channels
        )

        self.up_sample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder: save skip connections
        skip1 = self.inc(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        skip4 = self.down3(skip3)

        # Bottleneck
        x = self.bottleneck(skip4)  # [B, 512, 8, 8]

        # Decoder: upsample → skip connection (cat) → DoubleConv
        x = self.up_sample(x)  # [B, 512, 16, 16]
        x = torch.cat([skip4, x], dim=1)  # ── skip connection ──
        x = self.up_conv1(x)  # [B, 256, 16, 16]

        x = self.up_sample(x)  # [B, 256, 32, 32]
        x = torch.cat([skip3, x], dim=1)  # ── skip connection ──
        x = self.up_conv2(x)  # [B, 128, 32, 32]

        x = self.up_sample(x)  # [B, 128, 64, 64]
        x = torch.cat([skip2, x], dim=1)  # ── skip connection ──
        x = self.up_conv3(x)  # [B, 64, 64, 64]

        x = self.up_sample(x)  # [B, 64, 128, 128]
        x = torch.cat([skip1, x], dim=1)  # ── skip connection ──
        x = self.up_conv4(x)  # [B, 32, 128, 128]

        return self.out_conv(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net forward pass demo")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_channels", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--output_path", type=str, default="unet_result.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(args.in_channels, args.out_channels, args.hidden_channels).to(device)
    model.eval()

    x = torch.randn(
        args.batch_size, args.in_channels, args.height, args.width, device=device
    )

    with torch.no_grad():
        y = model(x)

    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(y.shape)}")

    def normalize(img):
        img = img - img.min()
        img = img / img.max()
        return img

    x_vis = normalize(x[0].cpu())
    y_vis = normalize(y[0].cpu())

    def to_visual(img):
        arr = img.permute(1, 2, 0).clamp(0, 1)
        if arr.shape[2] == 1:
            arr = arr.squeeze(-1)
        elif arr.shape[2] == 2:
            arr = arr[:, :, 0]
        elif arr.shape[2] > 3:
            arr = arr[:, :, :3]
        return arr.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(to_visual(x_vis), cmap="gray" if args.in_channels == 1 else None)
    axes[0].set_title("Input (random noise)")
    axes[0].axis("off")
    axes[1].imshow(to_visual(y_vis), cmap="gray" if args.out_channels == 1 else None)
    axes[1].set_title("Output")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=150)
    print(f"Saved visualization to {args.output_path}")
