"""
WillNet architecture for MRI super-resolution.

This module implements a simple convolutional neural network with a skip connection
for super-resolution of MRI images.
"""

import torch
import torch.nn as nn

# TODO: Review the model architecture and make improvements if necessary. 
# TODO: Add more comments and docstrings to explain the model's purpose and functionality.
class WillNet(nn.Module):
    """
    A simple super-resolution network with three convolutional layers and a skip connection.
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        out_channels (int): Number of output channels (default: 1 for grayscale)
        features (list): Number of features in each layer [64, 32, 1]
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 32, 1]):
        super(WillNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features[0], kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=features[0], out_channels=features[1], kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=features[1], out_channels=out_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
         # ─── NEW: small residual block ──────────
        self.resblock = nn.Sequential(
            nn.Conv2d(features[1], features[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[1], features[1], kernel_size=3, padding=1),
        )
        # ────────────────────────────────────────

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input low-resolution image (already upsampled)
            
        Returns:
            torch.Tensor: Super-resolution output image
        """
        residual = x 
        x = self.relu(self.conv1(x))  
        x = self.relu(self.conv2(x))  
        # ─── NEW residual block ───
        rb_in = x
        x    = self.resblock(x)
        x   += rb_in                  # local skip inside the block
        x    = self.relu(x)
        # ──────────────────────────
        x = x + residual 
        x = self.relu(self.conv3(x)) 
        return x


if __name__ == "__main__":
    # TODO: Add a test for the model
    model = WillNet()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")