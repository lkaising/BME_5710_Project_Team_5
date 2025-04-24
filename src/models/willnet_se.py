"""
WillNetSE architecture for MRI super-resolution.

This module implements a convolutional neural network with residual blocks
and channel attention (Squeeze-and-Excitation) for super-resolution of MRI images.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel-attention block"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)   # avoid collapse
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),             # (B,C,H,W) ➜ (B,C,1,1)
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)                    # channel-wise scaling


class WillNetSE(nn.Module):
    """
    An enhanced super-resolution network with residual blocks and channel attention.
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        out_channels (int): Number of output channels (default: 1 for grayscale)
        features (list): Number of features in each layer [64, 32, 1]
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 32, 1]):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, features[0], kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(features[0], features[1], kernel_size=5, padding=2)
        self.resblock = nn.Sequential(
            nn.Conv2d(features[1], features[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[1], features[1], kernel_size=3, padding=1)
        )
        self.se = SEBlock(features[1])
        self.conv3 = nn.Conv2d(features[1], out_channels, kernel_size=5, padding=2)

    def forward(self, x):
        """Forward pass"""
        residual = x                           # global skip (1-channel)

        x = self.relu(self.conv1(x))           # k=9
        x = self.relu(self.conv2(x))           # k=5

        # local SE-resblock
        rb_in = x
        x = self.resblock(x) + rb_in           # local skip
        x = self.relu(x)
        x = self.se(x)

        x = self.conv3(x)                      # 32→1  (no activation!)
        x = x + residual                       # channels now match

        return x                               # leave range unclipped


if __name__ == "__main__":
    # Test for the model
    model = WillNetSE()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")