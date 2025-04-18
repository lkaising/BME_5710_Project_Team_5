"""
Loss functions for super-resolution models.

This module implements various loss functions used in training
super-resolution models, including pixel-wise losses and perceptual losses.
"""

import torch
import torch.nn as nn
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def mse_loss(pred, target):
    """
    Mean Squared Error loss.
    
    Args:
        pred (torch.Tensor): Predicted image
        target (torch.Tensor): Target high-resolution image
        
    Returns:
        torch.Tensor: MSE loss
    """
    return nn.MSELoss()(pred, target)


def ssim_loss(pred, target, data_range=1.0):
    """
    Structural Similarity Index (SSIM) loss.
    
    Args:
        pred (torch.Tensor): Predicted image
        target (torch.Tensor): Target high-resolution image
        data_range (float): Data range of the images
        
    Returns:
        torch.Tensor: 1 - SSIM (as a loss function)
    """
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range).to(pred.device)
    ssim_val: torch.Tensor = ssim(pred, target)
    return 1.0 - ssim_val


def combined_loss(pred, target, gamma, data_range=1.0):
    """
    Combined MSE and SSIM loss.
    
    Args:
        pred (torch.Tensor): Predicted image
        target (torch.Tensor): Target high-resolution image
        gamma (float): Weight for MSE loss 
        data_range (float): Data range for SSIM calculation
        
    Returns:
        torch.Tensor: Combined loss
    """
    mse_loss_value = mse_loss(pred, target)
    ssim_loss_value = ssim_loss(pred, target, data_range)

    return gamma * mse_loss_value + (1.0 - gamma) * ssim_loss_value
