"""
Evaluation script for MRI super-resolution models.

This script evaluates trained models on test data and generates
detailed metrics and visualizations.
"""

import os
import argparse
import yaml
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from models import SRNET, WillNet
from utils import create_dataloaders, evaluate_metrics
from utils.visualization import plot_comparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MRI super-resolution model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--model", type=str, default="willnet", help="Model architecture: 'unet' or 'willnet'")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, model_name, device):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model_name (str): Name of the model architecture
        device (torch.device): Device to load the model on
        
    Returns:
        nn.Module: Loaded model
    """
    if model_name.lower() == 'unet':
        model = SRNET().to(device)
    elif model_name.lower() == 'willnet':
        model = WillNet().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'unet' or 'willnet'")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def evaluate_model(model, dataloader, device, output_dir):
    """
    Evaluate the model on the test set.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Test data loader
        device (torch.device): Device to use for evaluation
        output_dir (str): Directory to save results
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_metrics_interp = {'psnr': 0.0, 'ssim': 0.0}
    total_metrics_sr = {'psnr': 0.0, 'ssim': 0.0}
    
    visualize_batch_idx = len(dataloader) // 2
    visualize_sample_idx = 0
    
    with torch.no_grad():
        for batch_idx, (lr_img, hr_img) in enumerate(tqdm(dataloader, desc="Evaluating")):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            
            lr_img_up = F.interpolate(
                lr_img,
                scale_factor=2, 
                mode='bicubic',
                align_corners=False
            )
            
            sr_img = model(lr_img_up)
            
            metrics_interp = evaluate_metrics(lr_img_up, hr_img)
            metrics_sr = evaluate_metrics(sr_img, hr_img)
            
            for k in total_metrics_interp:
                total_metrics_interp[k] += metrics_interp[k]
                total_metrics_sr[k] += metrics_sr[k]
            
            if batch_idx == visualize_batch_idx:
                lr_np = lr_img_up[visualize_sample_idx].squeeze().cpu().numpy()
                hr_np = hr_img[visualize_sample_idx].squeeze().cpu().numpy()
                sr_np = sr_img[visualize_sample_idx].squeeze().cpu().numpy()
                
                fig = plot_comparison(lr_np, sr_np, hr_np, 
                                     save_path=os.path.join(output_dir, "sample_comparison.png"))
                
                error_interp = np.abs(hr_np - lr_np) * 5 
                error_sr = np.abs(hr_np - sr_np) * 5  
                
                np.save(os.path.join(output_dir, "sample_lr.npy"), lr_np)
                np.save(os.path.join(output_dir, "sample_hr.npy"), hr_np)
                np.save(os.path.join(output_dir, "sample_sr.npy"), sr_np)
                np.save(os.path.join(output_dir, "sample_error_interp.npy"), error_interp)
                np.save(os.path.join(output_dir, "sample_error_sr.npy"), error_sr)
    
    num_batches = len(dataloader)
    avg_metrics_interp = {k: v / num_batches for k, v in total_metrics_interp.items()}
    avg_metrics_sr = {k: v / num_batches for k, v in total_metrics_sr.items()}
    
    improvement = {
        'psnr': avg_metrics_sr['psnr'] - avg_metrics_interp['psnr'],
        'ssim': avg_metrics_sr['ssim'] - avg_metrics_interp['ssim']
    }
    
    results = {
        'interpolated': avg_metrics_interp,
        'super_resolved': avg_metrics_sr,
        'improvement': improvement
    }
    
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_model(args.checkpoint, args.model, device)
    print(f"Model loaded from {args.checkpoint}")
    
    data_dir = Path(config.get("data_dir", "./data"))
    transforms_config = config.get("transforms_config", "./configs/transforms.yaml")
    
    val_loader = create_dataloaders(
        data_dir=str(data_dir),
        config_path=transforms_config,
        loader_to_create="val",
        batch_size=1,
        num_workers=config.get("num_workers", 4)
    )
    
    results = evaluate_model(model, val_loader, device, str(output_dir))
    
    print("\nEvaluation Results:")
    print(f"Interpolated - PSNR: {results['interpolated']['psnr']:.2f} dB, SSIM: {results['interpolated']['ssim']:.4f}")
    print(f"Super-Resolved - PSNR: {results['super_resolved']['psnr']:.2f} dB, SSIM: {results['super_resolved']['ssim']:.4f}")
    print(f"Improvement - PSNR: {results['improvement']['psnr']:.2f} dB, SSIM: {results['improvement']['ssim']:.4f}")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()