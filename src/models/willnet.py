"""
WillNet architecture for MRI super-resolution.

This module implements a simple convolutional neural network with a skip connection
for super-resolution of MRI images.
"""

import torch
import torch.nn as nn

class WillNet(nn.Module):
    """
    A super-resolution network with multiple convolutional layers and skip connections.
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        out_channels (int): Number of output channels (default: 1 for grayscale)
        features (list): Number of features in each layer
    """
    
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 128, 64, 32, 1]):
        super(WillNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features[0], kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(in_channels=features[0], out_channels=features[1], kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=features[1], out_channels=features[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=features[2], out_channels=features[3], kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=features[3], out_channels=features[4], kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=features[4], out_channels=features[5], kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=features[5], out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

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
        x = self.relu(self.conv3(x))  
        x = self.relu(self.conv4(x))  
        x = self.relu(self.conv5(x))  
        x = self.relu(self.conv6(x))  
        x = x + residual 
        x = self.relu(self.conv7(x)) 
        return x


if __name__ == "__main__":
    model = WillNet()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
