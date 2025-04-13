"""Metrics for evaluating super-resolution models.

This module provides functions for calculating various image quality metrics
including PSNR, SSIM, and others for comparing image pairs.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def calculate_psnr(
    img1: Union[torch.Tensor, np.ndarray], 
    img2: Union[torch.Tensor, np.ndarray], 
    data_range: Optional[float] = None
) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image (predicted or generated).
        img2: Second image (ground truth).
        data_range: Data range of the images. If None, computed 
                    from the ground truth image.
        
    Returns:
        PSNR value in dB.
    """
    img1, img2, data_range = _prepare_metric_calculation(img1, img2, data_range)
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(img1.device)
    return psnr_metric(img1, img2).item()


def calculate_ssim(
    img1: Union[torch.Tensor, np.ndarray], 
    img2: Union[torch.Tensor, np.ndarray], 
    data_range: Optional[float] = None
) -> float:
    """Calculate Structural Similarity Index between two images.
    
    Args:
        img1: First image (predicted or generated).
        img2: Second image (ground truth).
        data_range: Data range of the images. If None, computed
                    from the ground truth image.
        
    Returns:
        SSIM value.
    """
    img1, img2, data_range = _prepare_metric_calculation(img1, img2, data_range)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(img1.device)
    return ssim_metric(img1, img2).item()


def evaluate_metrics(
    sr_images: Union[torch.Tensor, np.ndarray], 
    hr_images: Union[torch.Tensor, np.ndarray]
) -> Dict[str, float]:
    """Calculate multiple metrics for a batch of images.
    
    Args:
        sr_images: Super-resolution images.
        hr_images: High-resolution ground truth images.
        
    Returns:
        Dictionary containing metric values.
    """
    metrics = {
        'psnr': calculate_psnr(sr_images, hr_images),
        'ssim': calculate_ssim(sr_images, hr_images)
    }
    
    return metrics


def _prepare_metric_calculation(
    img1: Union[torch.Tensor, np.ndarray], 
    img2: Union[torch.Tensor, np.ndarray], 
    data_range: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Prepare images for metric calculation by ensuring they have correct format.
    
    Args:
        img1: First image.
        img2: Second image.
        data_range: Data range of the images. If None, computed 
                    from the ground truth image.
        
    Returns:
        Tuple of (img1, img2, data_range) prepared for metric calculation.
        
    Raises:
        AssertionError: If images don't have the same shape.
    """
    img1 = _prepare_image(img1)
    img2 = _prepare_image(img2)
    
    assert img1.shape == img2.shape, (
        f"Images must have the same shape: {img1.shape} vs {img2.shape}"
    )
    
    if data_range is None:
        data_range = img2.max() - img2.min()
        
    return img1, img2, data_range


def _prepare_image(img: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Convert input image to a torch.Tensor with proper dimensions and type.
        
    Args:
        img: Input image.
        
    Returns:
        Prepared image tensor.
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
    import time
    import argparse
    from pathlib import Path
    import torch.nn.functional as F
    
    parser = argparse.ArgumentParser(description="Test metric calculations on validation samples")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes")
    args = parser.parse_args()
    
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    data_dir = project_root / "data"
    
    start_time = time.time()
    
    print("\n" + "="*50)
    print("INTERPOLATED IMAGE QUALITY METRICS TEST")
    print("="*50)
    print(f"Data directory: {data_dir}")
    print("-"*50)
    
    try:
        from data import create_dataloaders
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        val_loader = create_dataloaders(
            data_dir=str(data_dir),
            loader_to_create='val',
            batch_size=args.batch_size, 
            num_workers=args.num_workers
        )
        
        total_metrics = {'psnr': 0.0, 'ssim': 0.0}
        total_samples = len(val_loader)
        
        print(f"\nProcessing {total_samples} validation samples...")
        
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
        
        print("\nAverage Interpolated Image Quality Metrics:")
        print(f"  • PSNR: {avg_metrics['psnr']:.2f} dB")
        print(f"  • SSIM: {avg_metrics['ssim']:.4f}")
        
        elapsed_time = time.time() - start_time
        print(f"\nMetrics calculation completed in {elapsed_time:.2f} seconds")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\nError during metrics test: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())
        print("="*50 + "\n")
        raise