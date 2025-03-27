"""
Data utilities for MRI super-resolution.

This module provides dataset classes and data loading utilities
for processing MRI image pairs.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from PIL import Image
import random


class MRIDataset(Dataset):
    """
    Dataset for loading pairs of low and high resolution MRI images.
    
    Args:
        lr_paths (list): List of paths to low-resolution images
        hr_paths (list): List of paths to high-resolution images
        transform (callable, optional): Optional transform to be applied on samples
    """
    
    def __init__(self, lr_paths, hr_paths, transform=None):
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.transform = transform
        
    def __len__(self):
        """Return the number of image pairs in the dataset."""
        return len(self.lr_paths)
    
    def __getitem__(self, idx):
        """
        Get the image pair at the specified index.
        
        Args:
            idx (int): Index of the image pair to fetch
            
        Returns:
            tuple: (lr_image, hr_image) tensors
        """
        lr_img = Image.open(self.lr_paths[idx])
        hr_img = Image.open(self.hr_paths[idx])

        lr_img = lr_img.resize((128, 128), Image.BICUBIC)
        
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        
        return lr_img, hr_img


def create_dataloaders(data_dir, batch_size=8, num_workers=4):
    """
    Create train and validation data loaders from pre-split directories.
    
    Args:
        data_dir (str): Root directory containing train and val folders
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_loader = _create_dataloader_from_dir(
        os.path.join(data_dir, 'train'),
        batch_size,
        num_workers,
        transform,
        shuffle=True
    )
    
    val_loader = _create_dataloader_from_dir(
        os.path.join(data_dir, 'val'),
        batch_size,
        num_workers,
        transform,
        shuffle=False
    )
    
    return train_loader, val_loader


def _create_dataloader_from_dir(split_dir, batch_size, num_workers, transform, shuffle=False):
    """Create a dataloader from a directory containing high-res and low-res subdirectories."""
    lr_paths, hr_paths = _get_image_paths_from_split(split_dir)
    
    dataset = MRIDataset(lr_paths, hr_paths, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def _get_image_paths_from_split(split_dir):
    """Get paired image paths from a split directory with high-res and low-res subdirs."""
    hr_dir = os.path.join(split_dir, 'high-res')
    lr_dir = os.path.join(split_dir, 'low-res')
    
    filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith('.tif')])
    
    hr_paths = [os.path.join(hr_dir, f) for f in filenames]
    lr_paths = [os.path.join(lr_dir, f) for f in filenames]
    
    return lr_paths, hr_paths