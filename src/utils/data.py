"""
Data utilities for MRI super-resolution.

This module provides dataset classes and data loading utilities
for processing MRI image pairs.
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class _MRIDataset(Dataset):
    """
    Dataset for loading pairs of low and high resolution MRI images.
    
    Args:
        lr_paths (list): List of paths to low-resolution images.
        hr_paths (list): List of paths to high-resolution images.
        transform (callable, optional): Optional transform to be applied on samples.
        lr_size (tuple, optional): Size for low-resolution images. Defaults to (128, 128).
        hr_size (tuple, optional): Size for high-resolution images. Defaults to (256, 256).
    """
    
    def __init__(self, lr_paths, hr_paths, transform=None, lr_size=(128, 128), hr_size=(256, 256)):
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.transform = transform
        self.lr_size = lr_size
        self.hr_size = hr_size
        
    def __len__(self):
        """Return the number of image pairs in the dataset."""
        return len(self.lr_paths)
    
    def __getitem__(self, idx):
        """
        Get the image pair at the specified index.
        
        Args:
            idx (int): Index of the image pair to fetch.
            
        Returns:
            tuple: (lr_image, hr_image) tensors.
        """
        lr_img = Image.open(self.lr_paths[idx])
        hr_img = Image.open(self.hr_paths[idx])

        lr_img = _ensure_size(lr_img, self.lr_size)
        hr_img = _ensure_size(hr_img, self.hr_size)

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        
        return lr_img, hr_img


def create_dataloaders(data_dir, loaders_to_create='both', batch_size=8, num_workers=4):
    """
    Create train and validation data loaders from pre-split directories.
    
    Args:
        data_dir (str): Root directory containing train and val folders.
        loaders_to_create (str, optional): Which loaders to create. Options are 
                                          'train', 'val', or 'both'. Defaults to 'both'.
        batch_size (int, optional): Batch size for training. Defaults to 8.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        
    Returns:
        If loaders_to_create is 'both': tuple of (train_loader, val_loader)
        If loaders_to_create is 'train': train_loader only
        If loaders_to_create is 'val': val_loader only
    
    Raises:
        ValueError: If loaders_to_create is not one of 'train', 'val', or 'both'.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    loaders = {}
    
    if loaders_to_create in ['train', 'both']:
        loaders['train'] = _create_dataloader_from_dir(
            os.path.join(data_dir, 'train'),
            batch_size,
            num_workers,
            transform,
            shuffle=True
        )
    
    if loaders_to_create in ['val', 'both']:
        loaders['val'] = _create_dataloader_from_dir(
            os.path.join(data_dir, 'val'),
            batch_size,
            num_workers,
            transform,
            shuffle=False
        )
    
    return_mapping = {
        'both': lambda: (loaders['train'], loaders['val']),
        'train': lambda: loaders['train'],
        'val': lambda: loaders['val']
    }
    
    try:
        return return_mapping[loaders_to_create]()
    except KeyError:
        raise ValueError("loaders_to_create must be one of 'train', 'val', or 'both'")


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
    
    print("\n" + "="*50)
    print("MRI DATASET TEST")
    print("="*50)
    print(f"Data directory: {data_dir}")
    print("-"*50)

    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0
    )
    
    print("Dataset statistics:")
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