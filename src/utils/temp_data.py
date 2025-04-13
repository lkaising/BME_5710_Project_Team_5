"""
Data utilities for MRI super-resolution.

This module provides dataset classes and data loading utilities
for processing MRI image pairs with configurable transformations.
"""

import os
import random
from PIL import Image
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class _MRIDataset(Dataset):
    """
    Dataset for loading paired low- and high-resolution MRI images.

    Args:
        lr_paths (list): Paths to low-resolution MRI images.
        hr_paths (list): Paths to high-resolution MRI images.
        transform (callable, optional): Transform applied jointly to the image pair.
        lr_size (tuple, optional): Target size for low-resolution images. Default: (128, 128)
        hr_size (tuple, optional): Target size for high-resolution images. Default: (256, 256)
    """

    def __init__(self, lr_paths, hr_paths, transform=None, lr_size=(128, 128), hr_size=(256, 256)):
        """Initialize the dataset with image paths and optional transforms."""
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
        Retrieve and transform the image pair at the specified index.

        Args:
            idx (int): Index of the image pair to fetch.

        Returns:
            tuple: (lr_image, hr_image) tensors after transformation.
        """
        lr_img = Image.open(self.lr_paths[idx])
        hr_img = Image.open(self.hr_paths[idx])

        def _ensure_size(img, target_size):
            """Resize image if it doesn't match the target size."""
            return img if img.size == target_size else img.resize(target_size, Image.BICUBIC)
        
        lr_img = _ensure_size(lr_img, self.lr_size)
        hr_img = _ensure_size(hr_img, self.hr_size)

        if self.transform:
            lr_img, hr_img = self.transform(lr_img, hr_img)

        return lr_img, hr_img


class _PairTransform:
    """
    Apply the same *geometric* transform to a low-res (LR) / high-res (HR)
    image pair, then apply (potentially different) *pixel* transforms.

    Parameters
    ----------
    geo_transforms : Optional[Callable]
        Albumentations transform that should be shared between LR & HR
        (e.g. flips, rotations).
    pixel_transforms_lr : Optional[Callable]
        Albumentations transform applied **only** to the LR image.
    pixel_transforms_hr : Optional[Callable]
        Albumentations transform applied **only** to the HR image.
        If ``None``, the LR pixel transform is reused.
    """

    def __init__(self, geo_transforms = None, pixel_transforms_lr = None, pixel_transforms_hr = None):
        """Initialize with optional transform pipelines."""
        self.geo_transforms = geo_transforms
        self.pixel_transforms_lr = pixel_transforms_lr
        self.pixel_transforms_hr = pixel_transforms_hr or pixel_transforms_lr

    def __call__(self, lr_img, hr_img):
        """
        Apply the same random transformation to both LR and HR images.

        Args:
            lr_img (PIL.Image.Image): Low-resolution image.
            hr_img (PIL.Image.Image): High-resolution image.

        Returns:
            tuple: Transformed (lr_tensor, hr_tensor) pair.
        """
        lr_arr, hr_arr = np.asarray(lr_img), np.asarray(hr_img)

        if self.geo_transforms:
            transformed = self.geo_transforms(image=lr_arr, image2=hr_arr)
            lr_arr, hr_arr = transformed["image"], transformed["image2"]

        seed = random.randint(0, 2**32)

        if self.pixel_transforms_lr:
            random.seed(seed)
            np.random.seed(seed)
            lr_tensor = self.pixel_transforms_lr(image=lr_arr)['image']
        else:
            lr_tensor = transforms.functional.to_tensor(lr_arr)

        if self.pixel_transforms_hr:
            random.seed(seed)
            np.random.seed(seed)
            hr_tensor = self.pixel_transforms_hr(image=hr_arr)["image"]
        else:
            hr_tensor = transforms.functional.to_tensor(hr_arr)

        return lr_tensor, hr_tensor

  
def create_dataloaders(data_dir, config_path, loader_to_create='both', batch_size=8, num_workers=4):
    """
    Build PyTorch dataloaders for train/val folders that each contain matching *low_res* / 
    *high_res* sub-directories.

    Parameters
    ----------
        data_dir : str | Path
            Root directory containing ``train`` and ``val`` sub-folders.
        config_path : str | Path
            YAML file describing the augmentation pipeline.
        loader_to_create : {`'train'`, `'val'`, `'both'`}, default `'both'`
            Which loader(s) to return.
        batch_size : int, default 8
        num_workers : int, default 4

    Returns
    -------
        DataLoader or tuple: 
            - If 'both': tuple of (train_loader, val_loader)
            - If 'train': train_loader only
            - If 'val': val_loader only

    Raises:
        ValueError: If loaders_to_create is not one of 'train', 'val', or 'both'.
    """
    valid_options = ['train', 'val', 'both']
    if loader_to_create not in valid_options:
        raise ValueError(
            f"loaders_to_create must be one of {valid_options}, got {loader_to_create}"
        )

    loaders = {}

    if loader_to_create in {'train', 'both'}:
        train_cfg = _read_transform_config(config_path, section='train')
        geo, px_lr, px_hr = _create_transforms_from_config(train_cfg)
        loaders['train'] = _create_dataloader_from_dir(
            os.path.join(data_dir, 'train'),
            batch_size, num_workers, _PairTransform(geo, px_lr, px_hr), shuffle=True
        )

    if loader_to_create in ['val', 'both']:
        val_cfg = _read_transform_config(config_path, section='val')
        geo, px_lr, px_hr = _create_transforms_from_config(val_cfg)
        loaders['val'] = _create_dataloader_from_dir(
            os.path.join(data_dir, 'val'),
            batch_size, num_workers, _PairTransform(geo, px_lr, px_hr), shuffle=False
        )

    return_mapping = {
        'both': lambda: (loaders['train'], loaders['val']),
        'train': lambda: loaders['train'],
        'val': lambda: loaders['val']
    }

    return return_mapping[loader_to_create]()


def _read_transform_config(config_path, section):
    """
    Read transformation configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        section (str): The section of the configuration to read.
            
    Returns:
        dict: Configuration dictionary with transform settings.
    """
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    global_transforms = cfg.get("transforms", {})

    if section not in cfg:
        raise ValueError(f"Section '{section}' not found in the configuration.")
    
    section_config = cfg[section]
    apply_transforms = section_config.get("apply_transforms", True)
    section_transforms = section_config.get("transforms") or {}

    if global_transforms is not None:
        combined_transforms = global_transforms.copy()
    else:
        combined_transforms = {}
    combined_transforms.update(section_transforms)

    return {"apply_transforms": apply_transforms, "transforms": combined_transforms}


def _create_transforms_from_config(config):
    """
    Build Albumentations transform pipelines from a config dict.

    Returns
    -------
    geo_transforms : Optional[Callable]
        Shared geometric transforms (LR & HR).
    pixel_transforms_lr : Optional[Callable]
        Pixel transforms for LR (includes GaussNoise if enabled).
    pixel_transforms_hr : Optional[Callable]
        Pixel transforms for HR (excludes GaussNoise).
    """
    if not config.get("apply_transforms", True):
        return None, None, None

    geo_transforms, px_lr, px_hr = [], [], []
    t_cfg = config.get("transforms", {})

    if t_cfg.get("horizontal_flip", {}).get("enabled", False):
        geo_transforms.append(
            A.HorizontalFlip(
                p=t_cfg["horizontal_flip"].get("p", 0.5)
            )
        )

    if t_cfg.get("vertical_flip", {}).get("enabled", False):
        geo_transforms.append(
            A.VerticalFlip(
                p=t_cfg["vertical_flip"].get("p", 0.5)
            )
        )

    if t_cfg.get("rotate", {}).get("enabled", False):
        geo_transforms.append(_create_rotation_transform(t_cfg.get("rotate", {})))

    if t_cfg.get("gauss_noise", {}).get("enabled", False):
        px_lr.append(
            A.GaussNoise(
                var_limit=tuple(t_cfg["gauss_noise"].get("var_limit", [0.001, 0.01])),
                p=t_cfg["gauss_noise"].get("p", 0.3),
            )
        )
    
    px_lr.append(ToTensorV2())
    px_hr.append(ToTensorV2())
    
    geo_comp = A.Compose(
        geo_transforms,
        additional_targets={'image2': 'image'},
        is_check_shapes=False
    ) if geo_transforms else None
    
    px_lr_comp = A.Compose(
        px_lr,
        is_check_shapes=False
    ) if px_lr else None

    px_hr_comp = A.Compose(
        px_hr,
        is_check_shapes=False
    ) if px_hr else None
    
    return geo_comp, px_lr_comp, px_hr_comp


def _create_rotation_transform(config):
    """Create rotation transform based on configuration."""
    mode = config.get("mode", "range")
    
    common_params = {
        "p": config.get("p", 0.5),
        "interpolation": config.get("interpolation", 1),
        "border_mode": config.get("border_mode", 4),
        "value": config.get("fill_value", None),
        "crop_border": config.get("crop_after", False)
    }
    
    if mode == "fixed":
        return _create_fixed_rotation(config, common_params)
    
    return _create_range_rotation(config, common_params)


def _create_fixed_rotation(config, common_params):
    """Create fixed angle rotation transform."""
    angles = config.get("fixed", {}).get("angles", [90, 180, 270])
    
    if set(angles).issubset({90, 180, 270}):
        return A.RandomRotate90(p=common_params["p"])
    
    rotations = [
        A.Rotate(limit=(angle, angle), p=1.0, **common_params) 
        for angle in angles
    ]
    return A.OneOf(rotations, p=common_params["p"])


def _create_range_rotation(config, common_params):
    """Create random range rotation transform."""
    limit = config.get("range", {}).get("limit", 15)
    return A.Rotate(
        limit=limit,
        mask_value=config.get("mask_value", None),
        **common_params
    )


def _create_dataloader_from_dir(split_dir, batch_size, num_workers, transform, shuffle=False):
    """
    Create a dataloader from a directory containing high-res and low-res subdirectories.
    
    Args:
        split_dir (str): Directory containing high-res and low-res subdirectories.
        batch_size (int): Batch size for the dataloader.
        num_workers (int): Number of workers for parallel loading.
        transform (callable): Transform to apply to the images.
        shuffle (bool, optional): Whether to shuffle the dataset. Default: False
        
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

    import matplotlib.pyplot as plt

    # Create a 2x2 grid for plotting both train and validation images
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot train sample images
    lr_img = lr_sample[0].squeeze(0).numpy()  
    hr_img = hr_sample[0].squeeze(0).numpy()

    axs[0, 0].imshow(lr_img, cmap='gray')
    axs[0, 0].set_title("Train Low Resolution")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(hr_img, cmap='gray')
    axs[0, 1].set_title("Train High Resolution")
    axs[0, 1].axis("off")

    # Get a sample batch from the validation loader and plot
    lr_sample_val, hr_sample_val = next(iter(val_loader))
    lr_img_val = lr_sample_val[0].squeeze(0).numpy()  
    hr_img_val = hr_sample_val[0].squeeze(0).numpy()

    axs[1, 0].imshow(lr_img_val, cmap='gray')
    axs[1, 0].set_title("Validation Low Resolution")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(hr_img_val, cmap='gray')
    axs[1, 1].set_title("Validation High Resolution")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
