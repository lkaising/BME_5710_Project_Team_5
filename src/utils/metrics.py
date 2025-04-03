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


def calculate_psnr(img1, img2):
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
    
    assert img1.shape == img2.shape, f"Images must have the same shape: {img1.shape} vs {img2.shape}"
    
    if img1.dim() == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    psnr_metric = PeakSignalNoiseRatio()
    psnr_metric = psnr_metric.to(img1.device)

    return psnr_metric(img1, img2).item()


def calculate_ssim(img1, img2, window_size=11):
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        window_size (int): Size of the SSIM window
        
    Returns:
        float: SSIM value
    """
    # Implementation will go here
    pass


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
    """Test PSNR calculation on sample data."""
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
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    val_dir = os.path.join(data_dir, 'val')
    lr_paths, hr_paths = _get_image_paths_from_split(val_dir)
    val_dataset = MRIDataset(lr_paths, hr_paths, transform=transform)
    psnr_metric = PeakSignalNoiseRatio().to(device)
    num_test_samples = min(5, len(val_dataset))
    
    print(f"\nTesting PSNR calculation on {num_test_samples} validation samples:")
    
    for i in range(num_test_samples):
        lr_img, hr_img = val_dataset[i]
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        
        # Upsample low-res image using bicubic interpolation
        lr_img_up = F.interpolate(
            lr_img.unsqueeze(0), 
            scale_factor=2, 
            mode='bicubic', 
            align_corners=False
        )
        hr_img = hr_img.unsqueeze(0)  # Add batch dimension
        
        # Calculate PSNR using our function
        psnr_value = calculate_psnr(lr_img_up, hr_img)
        
        # Calculate PSNR using torchmetrics directly for verification
        psnr_direct = psnr_metric(lr_img_up, hr_img).item()
        
        print(f"  Sample {i+1}:")
        print(f"    • PSNR (our function): {psnr_value:.2f} dB")
        print(f"    • PSNR (direct): {psnr_direct:.2f} dB")

    
    print("\nPSNR calculation test completed!")
    print("="*50 + "\n")