"""Data utilities for MRI super-resolution.

This module provides dataset classes and data loading utilities
for processing MRI image pairs with configurable transformations.
"""

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class _MRIDataset(Dataset):
    """Dataset for paired low-resolution and high-resolution MRI images.
    
    This dataset class loads paired MRI images for super-resolution tasks,
    ensuring proper sizing and applying optional transformations.
    
    Attributes:
        lr_paths: List of paths to low-resolution images.
        hr_paths: List of paths to high-resolution images.
        transform: Optional callable to transform the image pairs.
        lr_size: Target size for low-resolution images (width, height).
        hr_size: Target size for high-resolution images (width, height).
    """

    def __init__(
        self, 
        lr_paths: List[str], 
        hr_paths: List[str], 
        transform: Optional[Callable] = None, 
        lr_size: Tuple[int, int] = (128, 128), 
        hr_size: Tuple[int, int] = (256, 256)
    ):
        """Initialize the MRI dataset with image paths and transformations.
        
        Args:
            lr_paths: List of paths to low-resolution images.
            hr_paths: List of paths to high-resolution images.
            transform: Optional callable to transform the image pairs.
            lr_size: Target size for low-resolution images (width, height).
            hr_size: Target size for high-resolution images (width, height).
        """
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.transform = transform
        self.lr_size = lr_size
        self.hr_size = hr_size

    def __len__(self) -> int:
        """Return the number of image pairs in the dataset.
        
        Returns:
            Integer count of image pairs.
        """
        return len(self.lr_paths)

    def __getitem__(self, idx: int) -> Tuple:
        """Get a pair of LR and HR images at the specified index.
        
        Args:
            idx: Index of the image pair to retrieve.
            
        Returns:
            Tuple containing (low_resolution_image, high_resolution_image).
        """
        lr_img = Image.open(self.lr_paths[idx])
        hr_img = Image.open(self.hr_paths[idx])

        lr_img = self._ensure_size(lr_img, self.lr_size)
        hr_img = self._ensure_size(hr_img, self.hr_size)

        if self.transform:
            lr_img, hr_img = self.transform(lr_img, hr_img)

        return lr_img, hr_img

    @staticmethod
    def _ensure_size(img: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Ensure the image is at the target size, resizing if necessary.
        
        Args:
            img: PIL image to verify/resize.
            target_size: Desired dimensions (width, height).
            
        Returns:
            PIL image at the target size.
        """
        return img if img.size == target_size else img.resize(target_size, Image.BICUBIC)


class _PairTransform:
    """Transform utility for paired image processing.
    
    Applies consistent geometric transformations to both images
    and separate pixel-level transformations to each.
    
    Attributes:
        geo_transforms: Transformations applied to both images simultaneously.
        pixel_transforms_lr: Transformations applied to LR image only.
        pixel_transforms_hr: Transformations applied to HR image only.
    """

    def __init__(
        self, 
        geo_transforms: Optional[Callable] = None, 
        pixel_transforms_lr: Optional[Callable] = None, 
        pixel_transforms_hr: Optional[Callable] = None
    ):
        """Initialize transform with geometric and pixel-level transformations.
        
        Args:
            geo_transforms: Transformations applied to both images simultaneously.
            pixel_transforms_lr: Transformations applied to LR image only.
            pixel_transforms_hr: Transformations applied to HR image only.
                If None, uses the same transforms as pixel_transforms_lr.
        """
        self.geo_transforms = geo_transforms
        self.pixel_transforms_lr = pixel_transforms_lr
        self.pixel_transforms_hr = pixel_transforms_hr or pixel_transforms_lr

    def __call__(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple:
        """Apply transformations to the image pair.
        
        Args:
            lr_img: Low resolution PIL image.
            hr_img: High resolution PIL image.
            
        Returns:
            Tuple of (transformed_lr_tensor, transformed_hr_tensor).
        """
        lr_arr, hr_arr = np.asarray(lr_img).copy(), np.asarray(hr_img).copy()

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
    

def create_dataloaders(
    data_dir: Optional[str] = None, 
    config_path: Optional[str] = None,
    loader_to_create: Optional[str] = None, 
    batch_size: Optional[int] = 1, 
    num_workers: Optional[int] = 0
) -> Optional[Union[DataLoader, Tuple[DataLoader, DataLoader]]]:
    """Create train and/or validation dataloaders with appropriate transforms.
    
    Args:
        data_dir: Optional base directory containing train and val subdirectories.
            If None, no dataloaders are created.
        config_path: Optional path to YAML transform configuration file.
            If None, dataloaders are created without transformations.
        loader_to_create: Optional choice on which loader(s) to create ('train', 'val', or 'both').
            If None, no dataloaders are created.
        batch_size: Optional batch size for dataloaders.
            If None, defaults to 1.
        num_workers: Optional number of subprocesses for data loading.
            If None, defaults to 0.
        
    Returns:
        DataLoader or tuple of DataLoaders for training and validation.
        If data_dir is None or loader_to_create is None, returns None.
        
    Raises:
        ValueError: If loader_to_create is not one of 'train', 'val', or 'both'.
    """
    if data_dir is None:
        return None
    
    if loader_to_create is None:
        return None
    
    valid_options = ['train', 'val', 'both']
    if loader_to_create not in valid_options:
        raise ValueError(f"loader_to_create must be one of {valid_options}")
    
    loaders = {}
    
    if loader_to_create in ['train', 'both']:
        train_cfg = _read_transform_config(config_path, section='train')
        geo, px_lr, px_hr = _create_transforms_from_config(train_cfg)
        loaders['train'] = _create_dataloader_from_dir(
            str(Path(data_dir) / 'train'),
            batch_size, num_workers, 
            _PairTransform(geo, px_lr, px_hr), 
            shuffle=True
        )
    
    if loader_to_create in ['val', 'both']:
        val_cfg = _read_transform_config(config_path, section='val')
        geo, px_lr, px_hr = _create_transforms_from_config(val_cfg)
        loaders['val'] = _create_dataloader_from_dir(
            str(Path(data_dir) / 'val'),
            batch_size, num_workers, 
            _PairTransform(geo, px_lr, px_hr), 
            shuffle=False
        )
    
    return_mapping = {
        'both': lambda: (loaders['train'], loaders['val']),
        'train': lambda: loaders['train'],
        'val': lambda: loaders['val']
    }

    return return_mapping[loader_to_create]()


def _read_transform_config(
    config_path: Optional[str] = None, 
    section: str = "train"
) -> Optional[Dict]:
    """Read and merge transformation configuration from YAML file.
    
    Args:
        config_path: Optional path to YAML configuration file. If None, returns None.
        section: Section name to read (e.g., 'train', 'val').
        
    Returns:
        Dictionary with merged transformation configuration, or None if config_path is None.
        
    Raises:
        ValueError: If specified section is not found in config.
    """
    if config_path is None:
        return None
    
    config_data = yaml.safe_load(Path(config_path).read_text())
    
    try:
        section_data = config_data[section]
    except KeyError:
        raise ValueError(f"Section '{section}' not found in configuration.")
    
    global_settings = config_data.get("transforms") or {}
    section_settings = section_data.get("transforms") or {}
    
    combined_settings = {**global_settings, **section_settings}
    
    return {
        "apply_transforms": section_data.get("apply_transforms", True),
        "transforms": combined_settings
    }


def _create_rotation_transform(config: Dict) -> A.BasicTransform:
    """Create rotation transformation based on configuration.
    
    Args:
        config: Dictionary with rotation configuration parameters.
        
    Returns:
        Albumentations transform for rotation.
    """
    p = config.get("p", 0.5)
    common_params = {
        "p": p,
        "interpolation": config.get("interpolation", 1),
        "border_mode": config.get("border_mode", 4),
        "value": config.get("fill_value"),
        "crop_border": config.get("crop_after", False)
    }
    
    def _handle_fixed_rotation() -> A.BasicTransform:
        angles = config.get("fixed", {}).get("angles", [90, 180, 270])
        if set(angles).issubset({90, 180, 270}):
            return A.RandomRotate90(p=p)
        transforms = [A.Rotate(limit=(angle, angle), p=1.0, **common_params) for angle in angles]
        return A.OneOf(transforms, p=p)
    
    def _handle_range_rotation() -> A.BasicTransform:
        limit = config.get("range", {}).get("limit", 15)
        mask_value = config.get("mask_value")
        return A.Rotate(limit=limit, mask_value=mask_value, **common_params)
    
    mode = config.get("mode", "range")
    if mode == "fixed":
        return _handle_fixed_rotation()
    return _handle_range_rotation()


def _create_transforms_from_config(
    config: Optional[Dict] = None
) -> Tuple[Optional[A.Compose], Optional[A.Compose], Optional[A.Compose]]:
    """Create transformation pipelines from configuration.
    
    Args:
        config: Optional dictionary with transformation configuration.
            If None, returns (None, None, None).
        
    Returns:
        Tuple of (geometric_transforms, pixel_transforms_lr, pixel_transforms_hr).
        All elements are None if config is None or no transforms are specified.
    """
    if config is None:
        return None, None, None

    if not config.get("apply_transforms", True):
        return None, None, None

    transform_cfg = config.get("transforms", {})

    geo_transforms = []
    px_lr_transforms = []
    px_hr_transforms = []

    hflip_cfg = transform_cfg.get("horizontal_flip", {})
    if hflip_cfg.get("enabled", False):
        geo_transforms.append(A.HorizontalFlip(p=hflip_cfg.get("p", 0.5)))

    vflip_cfg = transform_cfg.get("vertical_flip", {})
    if vflip_cfg.get("enabled", False):
        geo_transforms.append(A.VerticalFlip(p=vflip_cfg.get("p", 0.5)))

    rotate_cfg = transform_cfg.get("rotate", {})
    if rotate_cfg.get("enabled", False):
        geo_transforms.append(_create_rotation_transform(rotate_cfg))

    noise_cfg = transform_cfg.get("gauss_noise", {})
    if noise_cfg.get("enabled", False):
        var_limit = tuple(noise_cfg.get("var_limit", [0.001, 0.01]))
        px_lr_transforms.append(A.GaussNoise(var_limit=var_limit, p=noise_cfg.get("p", 0.3)))

    px_lr_transforms.append(ToTensorV2())
    px_hr_transforms.append(ToTensorV2())

    geo_pipeline = A.Compose(
        geo_transforms, additional_targets={'image2': 'image'}, is_check_shapes=False
    ) if geo_transforms else None

    px_lr_pipeline = A.Compose(px_lr_transforms, is_check_shapes=False) if px_lr_transforms else None
    px_hr_pipeline = A.Compose(px_hr_transforms, is_check_shapes=False) if px_hr_transforms else None

    return geo_pipeline, px_lr_pipeline, px_hr_pipeline


def _get_image_paths_from_split(split_dir: str) -> Tuple[List[str], List[str]]:
    """Get paired image paths from a split directory.
    
    Args:
        split_dir: Directory containing 'high-res' and 'low-res' subdirectories.
        
    Returns:
        Tuple of (low_resolution_paths, high_resolution_paths).
    """
    split_path = Path(split_dir)
    hr_dir = split_path / 'high-res'
    lr_dir = split_path / 'low-res'
    
    filenames = sorted([f.name for f in hr_dir.glob('*.tif')])
    
    hr_paths = [str(hr_dir / f) for f in filenames]
    lr_paths = [str(lr_dir / f) for f in filenames]
    
    return lr_paths, hr_paths


def _create_dataloader_from_dir(
    split_dir: str, 
    batch_size: int, 
    num_workers: int, 
    transform: Optional[Callable] = None, 
    shuffle: bool = False
) -> DataLoader:
    """Create DataLoader from a directory containing image pairs.
    
    Args:
        split_dir: Directory containing 'high-res' and 'low-res' subdirectories.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        transform: Optional transform to apply to image pairs.
        shuffle: Whether to shuffle the dataset.
        
    Returns:
        DataLoader configured with the MRIDataset.
    """
    lr_paths, hr_paths = _get_image_paths_from_split(split_dir)
    
    dataset = _MRIDataset(lr_paths, hr_paths, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def _test_dataloaders(
    data_dir: Path, 
    config_path: Path, 
    batch_size: int = 4, 
    num_workers: int = 0,
    visualize: bool = True
):
    """Test MRIDataset and dataloader functionality.
    
    Args:
        data_dir: Directory containing train and val subdirectories
        config_path: Path to transforms configuration YAML file
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        visualize: Whether to display sample images
    """
    import time
    start_time = time.time()
    
    print("\n" + "="*50)
    print("MRI DATASET TEST")
    print("="*50)
    print(f"Data directory: {data_dir}")
    print(f"Config path: {config_path}")
    print("-"*50)
    
    try:
        train_loader, val_loader = create_dataloaders(
            data_dir=data_dir,
            loader_to_create='both',
            batch_size=batch_size,
            num_workers=num_workers,
            config_path=config_path
        )
        
        print("\nDataset statistics:")
        print(f"  • Training batches: {len(train_loader)}")
        print(f"  • Validation batches: {len(val_loader)}")
        
        lr_sample, hr_sample = next(iter(train_loader))
        lr_sample_val, hr_sample_val = next(iter(val_loader))
        
        print("\nSample batch:")
        print(f"  • Train shapes - LR: {lr_sample.shape}, HR: {hr_sample.shape}")
        print(f"  • Val shapes - LR: {lr_sample_val.shape}, HR: {hr_sample_val.shape}")
        print("  • Value ranges - ")
        print(f"    • Train LR: [{lr_sample.min():.3f}, {lr_sample.max():.3f}]")
        print(f"    • Train HR: [{hr_sample.min():.3f}, {hr_sample.max():.3f}]")
        print(f"    • Val LR: [{lr_sample_val.min():.3f}, {lr_sample_val.max():.3f}]")
        print(f"    • Val HR: [{hr_sample_val.min():.3f}, {hr_sample_val.max():.3f}]")
        
        elapsed_time = time.time() - start_time
        print(f"\nData successfully loaded in {elapsed_time:.2f} seconds")

        if visualize:
            _visualize_samples(lr_sample, hr_sample, lr_sample_val, hr_sample_val)
        
        print(f"\nTest completed!")
        print("="*50 + "\n")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"\nError during dataset test: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())
        print("="*50 + "\n")
        raise


def _visualize_samples(lr_train, hr_train, lr_val, hr_val):
    """Visualize sample images from training and validation sets.
    
    Args:
        lr_train: Low-resolution training samples
        hr_train: High-resolution training samples
        lr_val: Low-resolution validation samples
        hr_val: High-resolution validation samples
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        
        lr_img = lr_train[0].squeeze(0).numpy()  
        hr_img = hr_train[0].squeeze(0).numpy()
        
        axs[0, 0].imshow(lr_img, cmap='gray')
        axs[0, 0].set_title("Train Low Resolution")
        axs[0, 0].axis("off")
        
        axs[0, 1].imshow(hr_img, cmap='gray')
        axs[0, 1].set_title("Train High Resolution")
        axs[0, 1].axis("off")
        
        lr_img_val = lr_val[0].squeeze(0).numpy()  
        hr_img_val = hr_val[0].squeeze(0).numpy()
        
        axs[1, 0].imshow(lr_img_val, cmap='gray')
        axs[1, 0].set_title("Validation Low Resolution")
        axs[1, 0].axis("off")
        
        axs[1, 1].imshow(hr_img_val, cmap='gray')
        axs[1, 1].set_title("Validation High Resolution")
        axs[1, 1].axis("off")
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping visualization.")

  
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MRI dataset and dataloaders")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for dataloaders")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization")
    args = parser.parse_args()
    
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent
    data_dir = project_root / "data"
    config_path = project_root / "configs" / "transforms.yaml"
    
    _test_dataloaders(
        data_dir=data_dir,
        config_path=config_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        visualize=not args.no_viz
    )
