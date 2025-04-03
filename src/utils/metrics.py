"""
Metrics for evaluating super-resolution models.

This module provides functions for calculating various image quality metrics
including PSNR, SSIM, and others.
"""

import torch
import numpy
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def _prepare_image(img):
    """
    Convert input image to a torch.Tensor with proper dimensions and type.
        
    Args:
        img (np.ndarray or torch.Tensor): Input image.
        
    Returns:
        torch.Tensor: Prepared image tensor.
    """
    if isinstance(img, numpy.ndarray):
        img = torch.from_numpy(img)
    img = img.float()
    
    if img.dim() == 2:
        img = img.unsqueeze(0).unsqueeze(0)
    elif img.dim() == 3:
        img = img.unsqueeze(0)
    
    return img


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
    img1 = _prepare_image(img1)
    img2 = _prepare_image(img2)
    
    assert img1.shape == img2.shape, (
        f"Images must have the same shape: {img1.shape} vs {img2.shape}"
    )
    
    if data_range is None:
        data_range = img2.max() - img2.min()
    
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(img1.device)
    return psnr_metric(img1, img2).item()


def calculate_ssim(img1, img2, data_range=None):
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        img1 (torch.Tensor): First image.
        img2 (torch.Tensor): Second image.
        data_range (float, optional): Data range of the images. If None, computed
                                      from the ground truth image.
        
    Returns:
        float: SSIM value.
    """
    img1 = _prepare_image(img1)
    img2 = _prepare_image(img2)
    
    assert img1.shape == img2.shape, (
        f"Images must have the same shape: {img1.shape} vs {img2.shape}"
    )
    
    if data_range is None:
        data_range = img2.max() - img2.min()
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(img1.device)
    return ssim_metric(img1, img2).item()


def evaluate_metrics(sr_images, hr_images):
    """
    Calculate multiple metrics for a batch of images.
    
    Args:
        sr_images (torch.Tensor): Super-resolution images.
        hr_images (torch.Tensor): High-resolution ground truth images.
        
    Returns:
        dict: Dictionary containing metric values.
    """
    # TODO: Implement this function
    pass


def main():
    """Test PSNR and SSIM calculations on all validation samples."""
    import os
    import torch.nn.functional as F
    from torchvision import transforms
    from data import _get_image_paths_from_split, MRIDataset
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data")
    
    print("\n" + "="*50)
    print("INTERPOLATED IMAGE QUALITY METRICS TEST")
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
    
    total_psnr = 0.0
    total_ssim = 0.0
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
        
        psnr_value = calculate_psnr(lr_img_up, hr_img.unsqueeze(0))
        ssim_value = calculate_ssim(lr_img_up, hr_img.unsqueeze(0))
        
        total_psnr += psnr_value
        total_ssim += ssim_value
    
    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    
    print(f"Average Interpolated Image Quality Metrics:")
    print(f"  • PSNR: {avg_psnr:.2f} dB")
    print(f"  • SSIM: {avg_ssim:.4f}")
    
    print("\nMetrics calculation test completed!")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()