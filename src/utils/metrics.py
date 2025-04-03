#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics for evaluating super-resolution models.

This module provides functions for calculating various image quality metrics
including PSNR, SSIM, and others.
"""

import torch
import numpy as np
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def calculate_psnr(img1, img2, data_range=None):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1 (torch.Tensor or np.ndarray): First image.
        img2 (torch.Tensor or np.ndarray): Second image.
        data_range (float, optional): Data range of the images. If None, computed 
                                      from the ground truth image.
        
    Returns:
        float: PSNR value in dB.
    """
    img1, img2, data_range = _prepare_metric_calculation(img1, img2, data_range)
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(img1.device)
    return psnr_metric(img1, img2).item()


def calculate_ssim(img1, img2, data_range=None):
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        img1 (torch.Tensor or np.ndarray): First image.
        img2 (torch.Tensor or np.ndarray): Second image.
        data_range (float, optional): Data range of the images. If None, computed
                                      from the ground truth image.
        
    Returns:
        float: SSIM value.
    """
    img1, img2, data_range = _prepare_metric_calculation(img1, img2, data_range)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(img1.device)
    return ssim_metric(img1, img2).item()


def evaluate_metrics(sr_images, hr_images):
    """
    Calculate multiple metrics for a batch of images.
    
    Args:
        sr_images (torch.Tensor or np.ndarray): Super-resolution images.
        hr_images (torch.Tensor or np.ndarray): High-resolution ground truth images.
        
    Returns:
        dict: Dictionary containing metric values.
    """
    metrics = {
        'psnr': calculate_psnr(sr_images, hr_images),
        'ssim': calculate_ssim(sr_images, hr_images)
    }
    
    return metrics


def _prepare_metric_calculation(img1, img2, data_range=None):
    """
    Prepare images for metric calculation by ensuring they have correct format.
    
    Args:
        img1 (torch.Tensor or np.ndarray): First image.
        img2 (torch.Tensor or np.ndarray): Second image.
        data_range (float, optional): Data range of the images. If None, computed 
                                      from the ground truth image.
        
    Returns:
        tuple: (img1, img2, data_range) prepared for metric calculation.
    """
    img1 = _prepare_image(img1)
    img2 = _prepare_image(img2)
    
    assert img1.shape == img2.shape, (
        f"Images must have the same shape: {img1.shape} vs {img2.shape}"
    )
    
    if data_range is None:
        data_range = img2.max() - img2.min()
        
    return img1, img2, data_range


def _prepare_image(img):
    """
    Convert input image to a torch.Tensor with proper dimensions and type.
        
    Args:
        img (np.ndarray or torch.Tensor): Input image.
        
    Returns:
        torch.Tensor: Prepared image tensor.
    """
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    img = img.float()
    
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(0)
    
    return img


if __name__ == "__main__":
    """Test PSNR and SSIM calculations on all validation samples."""
    import os
    import torch.nn.functional as F
    from data import create_dataloaders
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data")
    
    print("\n" + "="*50)
    print("INTERPOLATED IMAGE QUALITY METRICS TEST")
    print("="*50)
    print(f"Data directory: {data_dir}")
    print("-"*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loader = create_dataloaders(
        data_dir=data_dir,
        loaders_to_create='val',
        batch_size=1, 
        num_workers=0
    )
    
    total_metrics = {'psnr': 0.0, 'ssim': 0.0}
    total_samples = len(val_loader)
    
    for lr_img, hr_img in val_loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        
        lr_img_up = F.interpolate(
            lr_img, 
            scale_factor=2, 
            mode='bicubic', 
            align_corners=False
        )
        
        metrics = evaluate_metrics(lr_img_up, hr_img)
        for key in total_metrics:
            total_metrics[key] += metrics[key]
    
    avg_metrics = {k: v / total_samples for k, v in total_metrics.items()}
    
    print("Average Interpolated Image Quality Metrics:")
    print(f"  • PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"  • SSIM: {avg_metrics['ssim']:.4f}")
    print("\nMetrics calculation test completed!")
    print("="*50 + "\n")