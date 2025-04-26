"""
Model definitions for MRI super-resolution.

This package contains neural network architectures for super-resolution
of MRI images, including U-Net variants and residual networks.
"""

from .willnet import WillNet
from .willnet_se_deep import WillNetSEDeep
from .trivialnet import TrivialNet
from .losses import mse_loss, ssim_loss, combined_loss