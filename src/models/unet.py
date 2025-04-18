"""
UNET architecture for MRI super-resolution.

This module implements a modified U-Net architecture with residual blocks
for super-resolution of MRI images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRNET(nn.Module):
    """
    Super-Resolution Network based on U-Net architecture.
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        out_channels (int): Number of output channels (default: 1 for grayscale)
        filters (list): Number of filters in each encoder/decoder block
    """
    
    def __init__(self, in_channels=1, out_channels=1, filters=[64, 128, 256, 512]):
        super(SRNET, self).__init__()
        
        # Encoder layers
        self.encoder1 = self.conv_block(in_channels, filters[0])
        self.encoder2 = self.conv_block(filters[0], filters[1])
        self.encoder3 = self.conv_block(filters[1], filters[2])
        self.encoder4 = self.conv_block(filters[2], filters[3])
        
        # Decoder layers
        self.decoder4 = self.upconv_block(filters[3], filters[2])
        self.decoder3 = self.upconv_block(filters[2], filters[1])
        self.decoder2 = self.upconv_block(filters[1], filters[0])
        self.decoder1 = self.upconv_block(filters[0], out_channels, final_layer=True)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def conv_block(self, in_channels, out_channels):
        """
        Convolutional block with two convolutional layers and ReLU activation.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            
        Returns:
            nn.Sequential: Convolutional block
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels, final_layer=False):
        """
        Upconvolutional block with one upconvolutional layer and ReLU activation.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            final_layer (bool): Whether this is the final layer (default: False)
            
        Returns:
            nn.Sequential: Upconvolutional block
        """
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        ]
        if not final_layer:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input low-resolution image
            
        Returns:
            torch.Tensor: Super-resolution output image
        """
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Decoder
        dec4 = self.decoder4(enc4)
        dec3 = self.decoder3(dec4 + enc3)
        dec2 = self.decoder2(dec3 + enc2)
        dec1 = self.decoder1(dec2 + enc1)
        
        return dec1


if __name__ == "__main__":
    # Quick test to verify model architecture
    model = SRNET()
    x = torch.randn(1, 1, 128, 128)  # Example input (batch, channels, height, width)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
