"""
Metrics for evaluating super-resolution models.

This module provides functions for calculating various image quality metrics
including PSNR, SSIM, and others.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def calculate_psnr(img1, img2, data_range=None):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1 (torch.Tensor or np.ndarray): First image
        img2 (torch.Tensor or np.ndarray): Second image
        
    Returns:
        float: PSNR value in dB
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    if not img1.is_floating_point():
        img1 = img1.float()
    if not img2.is_floating_point():
        img2 = img2.float()
    
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
    if img2.dim() == 2:
        img2 = img2.unsqueeze(0).unsqueeze(0)

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    assert img1.shape == img2.shape, f"Images must have the same shape: {img1.shape} vs {img2.shape}"

    if data_range is None:
        data_range = img2.max() - img2.min()
    
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range)
    psnr_metric = psnr_metric.to(img1.device)

    return psnr_metric(img1, img2).item()


def calculate_ssim(img1, img2, data_range=None):
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        data_range (float, optional): Data range of the images. If None, computed from the ground 
                                      truth image.
        
    Returns:
        float: SSIM value
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    if not img1.is_floating_point():
        img1 = img1.float()
    if not img2.is_floating_point():
        img2 = img2.float()
    
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
    if img2.dim() == 2:
        img2 = img2.unsqueeze(0).unsqueeze(0)

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    assert img1.shape == img2.shape, f"Images must have the same shape: {img1.shape} vs {img2.shape}"

    if data_range is None:
        data_range = img2.max() - img2.min()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range)
    ssim_metric = ssim_metric.to(img1.device)

    return ssim_metric(img1, img2).item()


def evaluate_metrics(sr_images, hr_images):
    """
    Calculate multiple metrics for a batch of images.
    
    Args:
        sr_images (torch.Tensor): Super-resolution images
        hr_images (torch.Tensor): High-resolution ground truth images
        
    Returns:
        dict: Dictionary containing metric values
    """
    # Implementation will go here
    pass


if __name__ == "__main__":
    """Test PSNR calculation on all validation samples."""
    import os
    from data import _get_image_paths_from_split, MRIDataset
    from torchvision import transforms
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data")
    
    print("\n" + "="*50)
    print(f"PSNR CALCULATION TEST")
    print("="*50)
    print(f"Data directory: {data_dir}")
    print("-"*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    val_dir = os.path.join(data_dir, 'val')
    lr_paths, hr_paths = _get_image_paths_from_split(val_dir)
    val_dataset = MRIDataset(lr_paths, hr_paths, transform=transform)
    
    psnr_metric = PeakSignalNoiseRatio().to(device)
    total_psnr_function = 0.0
    total_samples = len(val_dataset)
    
    
    for i in range(total_samples):
        lr_img, hr_img = val_dataset[i]
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        
        lr_img_up = F.interpolate(
            lr_img.unsqueeze(0), 
            scale_factor=2, 
            mode='bicubic', 
            align_corners=False
        )
        
        psnr_value = calculate_psnr(lr_img_up, hr_img)
        
        total_psnr_function += psnr_value
    
    avg_psnr_function = total_psnr_function / total_samples
    
    print(f"\nAverage PSNR results:")
    print(f"  • TOFDO UPDATE: {avg_psnr_function:.2f} dB")
    
    print("="*50 + "\n")