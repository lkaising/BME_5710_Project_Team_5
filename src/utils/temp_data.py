"""
Data utilities for MRI super-resolution.

This module provides dataset classes and data loading utilities
for processing MRI image pairs with configurable transformations.
"""


import os
import random
from PIL import Image
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def read_transform_config(config_path=None, *, section):
    """
    Reads transformation configuration from a YAML file.
    Args:
        ...
            
    Returns:
        ...
    """
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    global_transforms = cfg.get("transforms", {})

    if section not in cfg:
        raise ValueError(f"Section '{section}' not found in the configuration.")
    
    section_config = cfg[section]
    apply_transforms = section_config.get("apply_transforms", True)
    section_transforms = section_config.get("transforms") or {}

    combined_transforms = global_transforms.copy()
    combined_transforms.update(section_transforms)

    return {"apply_transforms": apply_transforms, "transforms": combined_transforms}


def create_transforms_from_config(config):
    """
    Creates Albumentations transforms based on the provided configuration.
    
    Args:
        config (dict): Configuration dictionary with transform settings.
        
    Returns:
        A.Compose: Composed transformations or None if transforms are disabled.
    """
    if not config.get('apply_transforms', True):
        return None
        
    transforms_list = []
    transforms_config = config.get('transforms', {})
    
    if transforms_config.get('horizontal_flip', {}).get('enabled', False):
        p = transforms_config.get('horizontal_flip', {}).get('p', 0.5)
        transforms_list.append(A.HorizontalFlip(p=p))
    
    if transforms_config.get('vertical_flip', {}).get('enabled', False):
        p = transforms_config.get('vertical_flip', {}).get('p', 0.5)
        transforms_list.append(A.VerticalFlip(p=p))
    
    if transforms_config.get('rotate', {}).get('enabled', False):
        limit = transforms_config.get('rotate', {}).get('limit', 15)
        p = transforms_config.get('rotate', {}).get('p', 0.5)
        transforms_list.append(A.Rotate(limit=limit, p=p))
    
    if transforms_config.get('elastic_transform', {}).get('enabled', False):
        alpha = transforms_config.get('elastic_transform', {}).get('alpha', 1.0)
        sigma = transforms_config.get('elastic_transform', {}).get('sigma', 50.0)
        alpha_affine = transforms_config.get('elastic_transform', {}).get('alpha_affine', 50.0)
        p = transforms_config.get('elastic_transform', {}).get('p', 0.5)
        transforms_list.append(A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, p=p))
    
    if transforms_config.get('gauss_noise', {}).get('enabled', False):
        var_limit = transforms_config.get('gauss_noise', {}).get('var_limit', [0.001, 0.01])
        p = transforms_config.get('gauss_noise', {}).get('p', 0.3)
        transforms_list.append(A.GaussNoise(var_limit=tuple(var_limit), p=p))
    
    if transforms_config.get('normalize', {}).get('enabled', True):
        mean = transforms_config.get('normalize', {}).get('mean', [0.5])
        std = transforms_config.get('normalize', {}).get('std', [0.5])
        transforms_list.append(A.Normalize(mean=tuple(mean), std=tuple(std)))
    
    transforms_list.append(ToTensorV2())
    
    return A.Compose(transforms_list) if transforms_list else None


class _PairTransform:
    """
    Applies the same transformations to both low-res and high-res images.

    Attributes:
        transforms (Callable): Albumentations transforms to apply to images.
    """

    def __init__(self, transforms=None):
        """
        Initializes the transform.

        Args:
            transforms (Callable, optional): Albumentations transforms to apply.
            normalize (bool, optional): Whether to normalize when no transforms are applied.
        """
        self.transforms = transforms

    def __call__(self, lr_img, hr_img):
        """
        Applies the same random transformation to both LR and HR images.

        Args:
            lr_img (PIL.Image.Image): Low-resolution image.
            hr_img (PIL.Image.Image): High-resolution image.

        Returns:
            Tuple[Tensor, Tensor]: Transformed (lr_tensor, hr_tensor).
        """
        if self.transforms:
            seed = random.randint(0, 2**32) 
            random.seed(seed)
            lr_tensor = self.transforms(image=np.array(lr_img))['image']
            random.seed(seed)
            hr_tensor = self.transforms(image=np.array(hr_img))['image']
        else:
            lr_tensor = transforms.functional.to_tensor(lr_img)
            hr_tensor = transforms.functional.to_tensor(hr_img)

        return lr_tensor, hr_tensor


class _MRIDataset(Dataset):
    """
    Dataset for loading paired low- and high-resolution MRI images.

    Attributes:
        lr_paths (list): Paths to low-resolution MRI images.
        hr_paths (list): Paths to high-resolution MRI images.
        transform (callable): A transform applied jointly to the image pair.
        lr_size (tuple): Target size for low-resolution images.
        hr_size (tuple): Target size for high-resolution images.
    """

    def __init__(self, lr_paths, hr_paths, transform=None, lr_size=(128, 128), hr_size=(256, 256)):
        """
        Initializes the dataset with lists of image paths and optional transforms.

        Args:
            lr_paths (list): List of paths to low-resolution images.
            hr_paths (list): List of paths to high-resolution images.
            transform (callable, optional): Optional transform applied to both images.
            lr_size (tuple, optional): Target size for low-resolution images. Defaults to (128, 128).
            hr_size (tuple, optional): Target size for high-resolution images. Defaults to (256, 256).
        """
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.transform = transform
        self.lr_size = lr_size
        self.hr_size = hr_size

    def __len__(self):
        """
        Returns the number of image pairs in the dataset.

        Returns:
            int: Number of paired samples.
        """
        return len(self.lr_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image pair at the specified index.

        Args:
            idx (int): Index of the image pair to fetch.

        Returns:
            tuple: (lr_image, hr_image) tensors after transformation.
        """
        lr_img = Image.open(self.lr_paths[idx])
        hr_img = Image.open(self.hr_paths[idx])

        lr_img = _ensure_size(lr_img, self.lr_size)
        hr_img = _ensure_size(hr_img, self.hr_size)

        if self.transform:
            lr_img, hr_img = self.transform(lr_img, hr_img)

        return lr_img, hr_img


def create_dataloaders(data_dir, loaders_to_create='both', batch_size=8, num_workers=4, config_path=None):
    """
    Creates train and/or validation data loaders from directory structure.

    Args:
        data_dir (str): Root directory containing 'train' and 'val' subfolders.
        loaders_to_create (str, optional): Which loaders to create. Must be one of
            'train', 'val', or 'both'. Defaults to 'both'.
        batch_size (int, optional): Batch size for loading data. Defaults to 8.
        num_workers (int, optional): Number of subprocesses for data loading. Defaults to 4.
        config_path (str, optional): Path to YAML configuration file. Defaults to None.

    Returns:
        tuple or DataLoader: 
            If loaders_to_create is 'both': returns (train_loader, val_loader)
            If loaders_to_create is 'train': returns train_loader
            If loaders_to_create is 'val': returns val_loader

    Raises:
        ValueError: If loaders_to_create is not one of 'train', 'val', or 'both'.
    """
    loaders = {}
    if loaders_to_create in ['train', 'both']:
        train_transform_config = read_transform_config(config_path, section='train')
        train_transforms = create_transforms_from_config(train_transform_config)
        train_transform = _PairTransform(transforms=train_transforms)
        loaders['train'] = _create_dataloader_from_dir(
            os.path.join(data_dir, 'train'),
            batch_size, num_workers, train_transform, shuffle=True
        )

    if loaders_to_create in ['val', 'both']:
        val_transform_config = read_transform_config(config_path, section='val')
        val_transforms = create_transforms_from_config(val_transform_config)
        val_transform = _PairTransform(transforms=val_transforms)
        loaders['val'] = _create_dataloader_from_dir(
            os.path.join(data_dir, 'val'),
            batch_size, num_workers, val_transform, shuffle=False
        )

    if loaders_to_create not in ['train', 'val', 'both']:
        raise ValueError(f"loaders_to_create must be one of 'train', 'val', or 'both', got {loaders_to_create}")

    return_mapping = {
        'both': lambda: (loaders['train'], loaders['val']),
        'train': lambda: loaders['train'],
        'val': lambda: loaders['val']
    }

    return return_mapping[loaders_to_create]()


def _ensure_size(img, target_size):
    """
    Resize image if it doesn't match target size.
    
    Args:
        img (PIL.Image): Image to resize.
        target_size (tuple): Target size as (width, height).
        
    Returns:
        PIL.Image: Resized image or original if already correct size.
    """
    return img if img.size == target_size else img.resize(target_size, Image.BICUBIC)


def _create_dataloader_from_dir(split_dir, batch_size, num_workers, transform, shuffle=False):
    """
    Create a dataloader from a directory containing high-res and low-res subdirectories.
    
    Args:
        split_dir (str): Directory containing high-res and low-res subdirectories.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for parallel loading.
        transform (callable): Transform to apply to the images.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        
    Returns:
        DataLoader: PyTorch dataloader for the dataset.
    """
    lr_paths, hr_paths = _get_image_paths_from_split(split_dir)
    
    dataset = _MRIDataset(lr_paths, hr_paths, transform=transform)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def _get_image_paths_from_split(split_dir):
    """
    Get paired image paths from a split directory with high-res and low-res subdirs.
    
    Args:
        split_dir (str): Directory containing high-res and low-res subdirectories.
        
    Returns:
        tuple: Lists of (lr_paths, hr_paths) with corresponding image pairs.
    """
    hr_dir = os.path.join(split_dir, 'high-res')
    lr_dir = os.path.join(split_dir, 'low-res')
    
    filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith('.tif')])
    
    hr_paths = [os.path.join(hr_dir, f) for f in filenames]
    lr_paths = [os.path.join(lr_dir, f) for f in filenames]
    
    return lr_paths, hr_paths


if __name__ == "__main__":
    """Test MRIDataset and dataloader functionality."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data")
    config_path = os.path.join(project_root, "configs", "transforms.yaml")
    
    print("\n" + "="*50)
    print("MRI DATASET TEST")
    print("="*50)
    print(f"Data directory: {data_dir}")
    print("-"*50)

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0,
        config_path=config_path
    )

    print("\nDataset statistics:")
    print(f"  • Training batches: {len(train_loader)}")
    print(f"  • Validation batches: {len(val_loader)}")
    
    lr_sample, hr_sample = next(iter(train_loader))

    print("\nSample batch:")
    print(f"  • Shapes - LR: {lr_sample.shape}, HR: {hr_sample.shape}")
    print("  • Value ranges - ")
    print(f"    • LR: [{lr_sample.min():.3f}, {lr_sample.max():.3f}]")
    print(f"    • HR: [{hr_sample.min():.3f}, {hr_sample.max():.3f}]")
    print("\nDataset test completed successfully!")
    print("="*50 + "\n")