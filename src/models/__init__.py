"""
Model definitions for MRI super-resolution.

This package contains neural network architectures for super-resolution
of MRI images, including U-Net variants and residual networks.
"""

from .unet import SRNET
from .resnet import ResidualBlock