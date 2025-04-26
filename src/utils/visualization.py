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


def plot_comparison(lr_img, sr_will, sr_triv, hr_img, save_path=None):
    """
    4-up image comparison + error maps with a shared colour-bar.
    """
    clip = lambda x: np.clip(
        (x.squeeze().cpu().numpy() if isinstance(x, torch.Tensor) else x), 0, 1
    )
    lr_img, sr_will, sr_triv, hr_img = map(clip, (lr_img, sr_will, sr_triv, hr_img))

    err_lr, err_will, err_triv = (5 * np.abs(hr_img - z) for z in (lr_img, sr_will, sr_triv))
    vmax = np.quantile(np.concatenate([err_lr.ravel(), err_will.ravel(), err_triv.ravel()]), 0.99)

    fig, ax = plt.subplots(
        2, 4, figsize=(16, 8),
        constrained_layout=True,
        gridspec_kw=dict(height_ratios=[1, 1])
    )

    titles = ["Ground Truth (HR)", "LR ↑ (bicubic)", "WillNet SR", "TrivialNet SR"]
    for i, (im, t) in enumerate(zip((hr_img, lr_img, sr_will, sr_triv), titles)):
        ax[0, i].imshow(im, cmap="gray", vmin=0, vmax=1)
        ax[0, i].set_title(t, fontsize=13, fontweight="bold")
        ax[0, i].axis("off")

    err_maps   = [np.zeros_like(hr_img), err_lr, err_will, err_triv]
    err_titles = ["zero error", "|HR−LR| ×5", "|HR−Will| ×5", "|HR−Triv| ×5"]

    for i, (e, t) in enumerate(zip(err_maps, err_titles)):
        im = ax[1, i].imshow(e, cmap="magma", vmin=0, vmax=vmax)
        ax[1, i].set_title(t, fontsize=11)
        ax[1, i].axis("off")

    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("absolute error ×5", rotation=90, labelpad=12)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

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