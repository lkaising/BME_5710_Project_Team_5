"""
WillNetSE-Plus architecture for MRI super-resolution.

This variant adds:
  •  Deeper stack of SE residual blocks (default = 8)
  •  Residual-scaling for stability (0.1 × block output)
  •  Learnable ×2 up-sampling (PixelShuffle) so the
     network consumes the *true* low-resolution image.
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-attention block"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


# ──────────────────────────────────────────────────────────────
class SEResBlock(nn.Module):
    """
    3×3-conv → ReLU → 3×3-conv → SE
    Residual scaling (0.1) keeps deeper nets stable without BN.
    """
    def __init__(self, channels: int, scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            SEBlock(channels)
        )
        self.scale = scale

    def forward(self, x):
        return x + self.scale * self.body(x)


# ──────────────────────────────────────────────────────────────
class WillNetSEPlus(nn.Module):
    """
    Deeper SR model: head → N×SE-ResBlocks → upsample → tail.

    Args:
        in_channels  (int):  # input channels (1 for MRI)
        out_channels (int):  # output channels (1 for MRI)
        mid_channels (int):  # width of the main trunk
        n_blocks     (int):  # how many SEResBlocks in the trunk
        upscale      (int):  up-scaling factor (2 or 4 typical)
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 mid_channels: int = 48,
                 n_blocks: int = 8,
                 upscale: int = 2):
        super().__init__()
        assert upscale in (2, 4), "PixelShuffle supports 2× or 4× here."

        # ---------- shallow feature extraction ----------
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 5, padding=2),
            nn.ReLU(inplace=True)
        )

        # ---------- deep feature extraction -------------
        self.body = nn.Sequential(
            *[SEResBlock(mid_channels) for _ in range(n_blocks)]
        )

        # ---------- up-sampling -------------------------
        self.upsampler = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale),
            nn.ReLU(inplace=True)
        )

        # ---------- reconstruction ----------------------
        self.tail = nn.Conv2d(mid_channels, out_channels, 3, padding=1)

    # ---------------------------------------------------
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.upsampler(x)
        x = self.tail(x)
        return x


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: LR 128×128 → HR 256×256 (×2)
    model = WillNetSEPlus(in_channels=1, out_channels=1, upscale=2)
    lr = torch.randn(1, 1, 128, 128)
    hr = model(lr)
    print(f"LR shape: {lr.shape}  ➜  HR shape: {hr.shape}")
