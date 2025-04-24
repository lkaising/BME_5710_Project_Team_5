"""
Loss functions for super-resolution models.

This module implements various loss functions used in training
super-resolution models, including pixel-wise losses and perceptual losses.
"""

import torch
import torch.nn as nn
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from utils import grad


def mse_loss(pred, target):
    # TODO: Update, check, and refactor the code. 
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
    # TODO: Update, check, and refactor the code. 
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


def charbonnier_loss(pred, target, eps=1e-6):
    return torch.sqrt((pred - target).pow(2) + eps).mean()


def combined_loss(pred, target, gamma, alpha=0.05, data_range=1.0):
    # TODO: Update, check, and refactor the code. 
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
    # mse_loss_value = mse_loss(pred, target)
    pix_lose_value = charbonnier_loss(pred, target)
    ssim_loss_value = ssim_loss(pred, target, data_range)
    edge_loss_value = charbonnier_loss(grad(pred), grad(target))

    return gamma  * pix_lose_value + (1.0 - gamma) * ssim_loss_value + alpha * edge_loss_value