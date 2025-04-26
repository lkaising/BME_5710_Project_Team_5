"""
Visualization utilities for MRI super-resolution.

This module provides functions for visualizing and saving results
of the super-resolution models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch


def plot_comparison(lr_img, sr_img, hr_img, save_path=None):
    """
    Plot low-res, super-res, and high-res images side by side with error maps.
    
    Args:
        lr_img (np.ndarray): Low-resolution image (upsampled to high-res size)
        sr_img (np.ndarray): Super-resolution image
        hr_img (np.ndarray): High-resolution ground truth image
        save_path (str, optional): Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure containing the plot
    """
    # Ensure inputs are numpy arrays
    if isinstance(lr_img, torch.Tensor):
        lr_img = lr_img.squeeze().cpu().numpy()
    if isinstance(sr_img, torch.Tensor):
        sr_img = sr_img.squeeze().cpu().numpy()
    if isinstance(hr_img, torch.Tensor):
        hr_img = hr_img.squeeze().cpu().numpy()
    
    # Make sure all values are in [0, 1] range
    lr_img = np.clip(lr_img, 0, 1)
    sr_img = np.clip(sr_img, 0, 1)
    hr_img = np.clip(hr_img, 0, 1)
    
    # Calculate error maps (scale by 5 for visibility)
    error_lr = 5 * np.abs(hr_img - lr_img)
    error_sr = 5 * np.abs(hr_img - sr_img)
    
    # Create plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot images
    ax[0, 0].imshow(hr_img, cmap='gray', vmin=0, vmax=1)
    ax[0, 0].set_title('Ground Truth (High-Res)')
    ax[0, 0].axis('off')
    
    ax[0, 1].imshow(lr_img, cmap='gray', vmin=0, vmax=1)
    ax[0, 1].set_title('Interpolated Low-Res')
    ax[0, 1].axis('off')
    
    ax[0, 2].imshow(sr_img, cmap='gray', vmin=0, vmax=1)
    ax[0, 2].set_title('Super-Resolved')
    ax[0, 2].axis('off')
    
    # Plot error maps
    ax[1, 0].imshow(np.zeros_like(hr_img), cmap='gray', vmin=0, vmax=1)
    ax[1, 0].set_title('Reference (Zero Error)')
    ax[1, 0].axis('off')
    
    ax[1, 1].imshow(error_lr, cmap='gray', vmin=0, vmax=1)
    ax[1, 1].set_title('Error: Ground Truth - Interpolated (×5)')
    ax[1, 1].axis('off')
    
    ax[1, 2].imshow(error_sr, cmap='gray', vmin=0, vmax=1)
    ax[1, 2].set_title('Error: Ground Truth - Super-Resolved (×5)')
    ax[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_results(images, output_dir, prefix="result"):
    """
    Save a batch of images to the specified directory.
    
    Args:
        images (torch.Tensor or np.ndarray): Batch of images to save
        output_dir (str): Directory to save images
        prefix (str): Prefix for image filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy if tensor
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # Handle different input shapes
    if images.ndim == 4:  # Batch of images [B, C, H, W]
        batch_size = images.shape[0]
        for i in range(batch_size):
            img = images[i]
            if img.shape[0] == 1:  # Single channel
                img = img.squeeze(0)
            
            # Ensure values are in [0, 1] range
            img = np.clip(img, 0, 1)
            
            # Convert to 8-bit
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            
            # Save image
            img_path = os.path.join(output_dir, f"{prefix}_{i:04d}.tif")
            img_pil.save(img_path)
    
    elif images.ndim == 3:  # Single image with channels [C, H, W] or batch of grayscale [B, H, W]
        if images.shape[0] == 1 or images.shape[0] == 3:  # Likely [C, H, W]
            img = images.transpose(1, 2, 0) if images.shape[0] == 3 else images.squeeze(0)
        else:  # Likely [B, H, W]
            for i in range(images.shape[0]):
                img = images[i]
                
                # Ensure values are in [0, 1] range
                img = np.clip(img, 0, 1)
                
                # Convert to 8-bit
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                
                # Save image
                img_path = os.path.join(output_dir, f"{prefix}_{i:04d}.tif")
                img_pil.save(img_path)
            return
        
        # Ensure values are in [0, 1] range
        img = np.clip(img, 0, 1)
        
        # Convert to 8-bit
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        # Save image
        img_path = os.path.join(output_dir, f"{prefix}.tif")
        img_pil.save(img_path)
    
    elif images.ndim == 2:  # Single grayscale image [H, W]
        # Ensure values are in [0, 1] range
        img = np.clip(images, 0, 1)
        
        # Convert to 8-bit
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        # Save image
        img_path = os.path.join(output_dir, f"{prefix}.tif")
        img_pil.save(img_path)
    
    else:
        raise ValueError(f"Unsupported image shape: {images.shape}")


def create_grid(images, nrow=8, padding=2):
    """
    Create a grid of images.
    
    Args:
        images (torch.Tensor or list): Images to put in the grid
        nrow (int): Number of images per row
        padding (int): Padding between images
        
    Returns:
        np.ndarray: Grid of images
    """
    # Convert list of tensors to a single tensor
    if isinstance(images, list):
        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        else:
            images = np.stack(images)
    
    # Convert torch tensor to numpy
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    # Get dimensions
    if images.ndim == 4:  # [B, C, H, W]
        batch_size, channels, height, width = images.shape
    elif images.ndim == 3 and (images.shape[0] == 1 or images.shape[0] == 3):  # [C, H, W]
        channels, height, width = images.shape
        batch_size = 1
        images = images.reshape(batch_size, channels, height, width)
    else:
        raise ValueError(f"Unsupported image shape: {images.shape}")
    
    # Compute grid dimensions
    ncol = min(nrow, batch_size)
    nrow = (batch_size + ncol - 1) // ncol
    
    # Create empty grid
    grid_height = nrow * height + (nrow - 1) * padding
    grid_width = ncol * width + (ncol - 1) * padding
    grid = np.zeros((channels, grid_height, grid_width), dtype=images.dtype)
    
    # Fill grid with images
    for idx in range(min(batch_size, nrow * ncol)):
        row = idx // ncol
        col = idx % ncol
        grid[
            :,
            row * (height + padding) : row * (height + padding) + height,
            col * (width + padding) : col * (width + padding) + width
        ] = images[idx]
    
    # Transpose for visualization [C, H, W] -> [H, W, C]
    if channels == 3:
        grid = grid.transpose(1, 2, 0)
    else:
        grid = grid.squeeze(0)
    
    return grid

# def show_sample(loader, title):
#     lr, hr = next(iter(loader))
#     plt.figure(figsize=(6,3))
#     plt.subplot(1,2,1); plt.imshow(lr[0,0], cmap='gray'); plt.title('LR'); plt.axis('off')
#     plt.subplot(1,2,2); plt.imshow(hr[0,0], cmap='gray'); plt.title('HR'); plt.axis('off')
#     plt.suptitle(title); plt.show()

def show_sample(batch, title):
    """
    Displays the low-resolution (LR) and high-resolution (HR) images
    from a given batch.

    Args:
        batch: A tuple or list containing (lr_tensor, hr_tensor).
               Assumes tensors have shape (batch_size, channels, height, width).
        title: The title for the plot window.
    """
    lr, hr = batch

    lr_img = lr[0].squeeze().detach().cpu().numpy()
    hr_img = hr[0].squeeze().detach().cpu().numpy()

    plt.figure(figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(lr_img, cmap='gray') 
    plt.title('Low Resolution (LR)', fontsize=12)
    plt.axis('off') 

    plt.subplot(1, 2, 2)
    plt.imshow(hr_img, cmap='gray') 
    plt.title('High Resolution (HR)', fontsize=12)
    plt.axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    plt.show()