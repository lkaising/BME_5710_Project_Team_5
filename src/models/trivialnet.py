#!/usr/bin/env python3
# trivialnet.py – a two-layer baseline that learns a 3×3 local filter

from __future__ import annotations
import torch.nn as nn

class TrivialNet(nn.Module):
    """
    A minimal baseline: 3×3 conv → ReLU → 3×3 conv.
    The network is deliberately shallow so it can be
    trained in a few minutes and used as a learned ‘identity’.
    """
    def __init__(self, in_channels: int = 1, mid: int = 12, out_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, out_channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))
