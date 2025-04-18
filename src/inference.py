"""
Inference script for MRI super-resolution models.

This script handles loading a trained model and performing
super-resolution on new low-resolution images.
"""

import os
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np

from models import SRNET, WillNet
from utils.visualization import save_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with MRI super-resolution model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--model", type=str, default="willnet", help="Model architecture: 'unet' or 'willnet'")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        print(f"Warning: Config file {config_path} is empty or invalid. Using default values.")
        config = {}
    
    return config


def load_model(checkpoint_path, model_name, device):
    """
    Load the model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        model_name (str): Model architecture name
        device (torch.device): Device to load the model on
        
    Returns:
        nn.Module: Loaded model
    """
    if model_name.lower() == 'unet':
        model = SRNET().to(device)
    elif model_name.lower() == 'willnet':
        model = WillNet().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'unet' or 'trivial'")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def preprocess_image(image_path):
    """
    Preprocess an image for inference.
    
    Args:
        image_path (str): Path to input image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Load image as grayscale
    img = Image.open(image_path).convert('L')
    
    # Convert to numpy array and normalize to [0, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Convert to tensor and add batch and channel dimensions
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    
    return img_tensor


def run_inference(model, input_dir, output_dir, device):
    """
    Run inference on all images in the input directory.
    
    Args:
        model (nn.Module): Model to use for inference
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save results
        device (torch.device): Device to use for inference
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Running inference on {len(image_files)} images...")
    
    for filename in tqdm(image_files, desc="Processing"):
        # Preprocess image
        input_path = os.path.join(input_dir, filename)
        img_tensor = preprocess_image(input_path)
        img_tensor = img_tensor.to(device)
        
        # Upsample the input
        img_upsampled = F.interpolate(
            img_tensor,
            scale_factor=2,
            mode='bicubic',
            align_corners=False
        )
        
        # Run inference
        with torch.no_grad():
            sr_output = model(img_upsampled)
        
        # Save both upsampled and super-resolved outputs
        upsampled_np = img_upsampled.squeeze().cpu().numpy()
        sr_np = sr_output.squeeze().cpu().numpy()
        
        # Ensure values are in [0, 1] range
        upsampled_np = np.clip(upsampled_np, 0, 1)
        sr_np = np.clip(sr_np, 0, 1)
        
        # Convert to 8-bit
        upsampled_img = Image.fromarray((upsampled_np * 255).astype(np.uint8))
        sr_img = Image.fromarray((sr_np * 255).astype(np.uint8))
        
        # Save images
        base_name = os.path.splitext(filename)[0]
        upsampled_path = os.path.join(output_dir, f"{base_name}_upsampled.tif")
        sr_path = os.path.join(output_dir, f"{base_name}_sr.tif")
        
        upsampled_img.save(upsampled_path)
        sr_img.save(sr_path)
    
    print(f"Inference completed. Results saved to {output_dir}")


def main():
    """Main inference function."""
    args = parse_args()
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = load_model(args.checkpoint, args.model, device)
    print(f"Model loaded from {args.checkpoint}")
    
    run_inference(model, args.input_dir, args.output_dir, device)


if __name__ == "__main__":
    main()