"""
Data utilities for MRI super-resolution.

This module provides dataset classes and data loading utilities
for processing MRI image pairs with configurable transformations.
"""

import os
import random
from typing import List, Tuple, Dict, Optional, Callable, Union
from PIL import Image
import numpy as np
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MRIDataset(Dataset):
    """
    ...
    """

    def __init__(
        self, 
        lr_paths: List[str], 
        hr_paths: List[str], 
        transform: Optional[Callable] = None, 
        lr_size: Tuple[int, int] = (128, 128), 
        hr_size: Tuple[int, int] = (256, 256)
    ):
        self.lr_paths = lr_paths
        self.hr_paths = hr_paths
        self.transform = transform
        self.lr_size = lr_size
        self.hr_size = hr_size

    def __len__(self) -> int:
        """
        ...
        """
        return len(self.lr_paths)

    def __getitem__(self, idx: int) -> Tuple:
        """
        ...
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
        """
        ...
        """
        return img if img.size == target_size else img.resize(target_size, Image.BICUBIC)


class PairTransform:
    """
    ....
    """

    def __init__(
        self, 
        geo_transforms: Optional[Callable] = None, 
        pixel_transforms_lr: Optional[Callable] = None, 
        pixel_transforms_hr: Optional[Callable] = None
    ):
        self.geo_transforms = geo_transforms
        self.pixel_transforms_lr = pixel_transforms_lr
        self.pixel_transforms_hr = pixel_transforms_hr or pixel_transforms_lr

    def __call__(self, lr_img: Image.Image, hr_img: Image.Image) -> Tuple:
        """
        ...
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
    

def read_transform_config(config_path: str, section: str) -> Dict:
    """
    ...
    """
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


def create_rotation_transform(config: Dict) -> A.BasicTransform:
    """
    ...
    """
    p = config.get("p", 0.5)
    common_params = {
        "p": p,
        "interpolation": config.get("interpolation", 1),
        "border_mode": config.get("border_mode", 4),
        "value": config.get("fill_value"),
        "crop_border": config.get("crop_after", False)
    }
    
    def handle_fixed_rotation() -> A.BasicTransform:
        angles = config.get("fixed", {}).get("angles", [90, 180, 270])
        if set(angles).issubset({90, 180, 270}):
            return A.RandomRotate90(p=p)
        transforms = [A.Rotate(limit=(angle, angle), p=1.0, **common_params) for angle in angles]
        return A.OneOf(transforms, p=p)
    
    def handle_range_rotation() -> A.BasicTransform:
        limit = config.get("range", {}).get("limit", 15)
        mask_value = config.get("mask_value")
        return A.Rotate(limit=limit, mask_value=mask_value, **common_params)
    
    mode = config.get("mode", "range")
    if mode == "fixed":
        return handle_fixed_rotation()
    return handle_range_rotation()


def create_transforms_from_config(
    config: Dict
) -> Tuple[Optional[A.Compose], Optional[A.Compose], Optional[A.Compose]]:
    """
    ...
    """
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
        geo_transforms.append(create_rotation_transform(rotate_cfg))

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


def get_image_paths_from_split(split_dir: str) -> Tuple[List[str], List[str]]:
    """
    ...
    """
    hr_dir = os.path.join(split_dir, 'high-res')
    lr_dir = os.path.join(split_dir, 'low-res')
    
    filenames = sorted([f for f in os.listdir(hr_dir) if f.endswith('.tif')])
    
    hr_paths = [os.path.join(hr_dir, f) for f in filenames]
    lr_paths = [os.path.join(lr_dir, f) for f in filenames]
    
    return lr_paths, hr_paths


def create_dataloader_from_dir(
    split_dir: str, 
    batch_size: int, 
    num_workers: int, 
    transform: Optional[Callable] = None, 
    shuffle: bool = False
) -> DataLoader:
    """
    ...
    """
    lr_paths, hr_paths = get_image_paths_from_split(split_dir)
    
    dataset = MRIDataset(lr_paths, hr_paths, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def create_dataloaders(
    data_dir: str, 
    config_path: str, 
    loader_to_create: str = 'both', 
    batch_size: int = 8, 
    num_workers: int = 4
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    ...
    """
    valid_options = ['train', 'val', 'both']
    if loader_to_create not in valid_options:
        raise ValueError(f"loader_to_create must be one of {valid_options}")
    
    loaders = {}
    
    if loader_to_create in ['train', 'both']:
        train_cfg = read_transform_config(config_path, section='train')
        geo, px_lr, px_hr = create_transforms_from_config(train_cfg)
        loaders['train'] = create_dataloader_from_dir(
            os.path.join(data_dir, 'train'),
            batch_size, num_workers, 
            PairTransform(geo, px_lr, px_hr), 
            shuffle=True
        )
    
    if loader_to_create in ['val', 'both']:
        val_cfg = read_transform_config(config_path, section='val')
        geo, px_lr, px_hr = create_transforms_from_config(val_cfg)
        loaders['val'] = create_dataloader_from_dir(
            os.path.join(data_dir, 'val'),
            batch_size, num_workers, 
            PairTransform(geo, px_lr, px_hr), 
            shuffle=False
        )
    
    return_mapping = {
        'both': lambda: (loaders['train'], loaders['val']),
        'train': lambda: loaders['train'],
        'val': lambda: loaders['val']
    }

    return return_mapping[loader_to_create]()


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

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    lr_img = lr_sample[0].squeeze(0).numpy()  
    hr_img = hr_sample[0].squeeze(0).numpy()

    axs[0, 0].imshow(lr_img, cmap='gray')
    axs[0, 0].set_title("Train Low Resolution")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(hr_img, cmap='gray')
    axs[0, 1].set_title("Train High Resolution")
    axs[0, 1].axis("off")

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