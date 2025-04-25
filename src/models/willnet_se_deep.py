"""
WillNetSE architecture for MRI super-resolution.

This module implements a deep neural network with Squeeze-and-Excitation blocks for 
super-resolution of MRI images.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp 


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
    
    Args:
        channels (int): Number of input channels
        reduction (int, optional): Reduction ratio for the hidden layer. Defaults to 8.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden_channels = max(1, channels // reduction)

        SE_KERNEL_SIZE = 1
        
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=channels,
                out_channels=hidden_channels,
                kernel_size=SE_KERNEL_SIZE,
                bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=channels,
                kernel_size=SE_KERNEL_SIZE,
                bias=True
            ),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SE block."""
        return x * self.conv(x)

   
class SEResBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation.
    
    Args:
        channels (int): Number of input and output channels
        scale (float, optional): Scaling factor for the residual path. Defaults to 0.1.
    """
    def __init__(self, channels: int, scale: float = 0.1):
        super().__init__()

        KERNEL_SIZE = 3
        STRIDE = 1
        PADDING = 1 

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                padding=PADDING
            ),
            SEBlock(channels)
        )
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SE residual block."""
        return x + self.scale * self.body(x)


class WillNetSEDeep(nn.Module):
    """
    Deep WillNet model with Squeeze-and-Excitation blocks for MRI super-resolution.
    
    The architecture consists of:
    1. A head network to extract initial features
    2. A body with a sequence of SE residual blocks
    3. A tail network to produce the final output
    4. A global residual connection from input to output
    
    Args:
        n_blocks (int, optional): Number of SE residual blocks. Defaults to 8.
        mid_channels (int, optional): Number of channels in the body. Defaults to 48.
    """
    def __init__(self, n_blocks: int = 8, mid_channels: int = 48):
        super().__init__()

        IN_CHANNELS = 1
        OUT_CHANNELS = 1
        INITIAL_FEATURES = 64 
        
        HEAD_FIRST_KERNEL = 9
        HEAD_FIRST_PADDING = 4
        HEAD_SECOND_KERNEL = 5
        HEAD_SECOND_PADDING = 2
        
        TAIL_KERNEL = 5
        TAIL_PADDING = 2 

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=IN_CHANNELS,
                out_channels=INITIAL_FEATURES,
                kernel_size=HEAD_FIRST_KERNEL,
                padding=HEAD_FIRST_PADDING
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=INITIAL_FEATURES,
                out_channels=mid_channels,
                kernel_size=HEAD_SECOND_KERNEL,
                padding=HEAD_SECOND_PADDING
            ),
            nn.ReLU(inplace=True)
        )

        self.body = nn.ModuleList([
            SEResBlock(mid_channels) for _ in range(n_blocks)
        ])

        self.tail = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=OUT_CHANNELS,
            kernel_size=TAIL_KERNEL,
            padding=TAIL_PADDING
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the WillNetSEDeep model.
        
        Uses gradient checkpointing to reduce memory usage during training.
        
        Args:
            x (torch.Tensor): Input MRI image tensor of shape [B, 1, H, W]
            
        Returns:
            torch.Tensor: Super-resolved output tensor of shape [B, 1, H, W]
        """
        res = x
        x = self.head(x)

        for blk in self.body:
            x = cp.checkpoint(blk, x, use_reentrant=False)

        x = self.tail(x)
        return x + res
