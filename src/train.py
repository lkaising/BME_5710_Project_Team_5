"""
Training script for MRI super-resolution models.

This script handles the training loop, checkpointing, and logging
for training super-resolution models on MRI data.
"""

import os
import argparse
import yaml
import time
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import SRNET, WillNet, WillNetSE, WillNetSEPlus
from models import combined_loss
from utils import create_dataloaders, evaluate_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MRI super-resolution model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model", type=str, default="willnet", help="Model architecture: 'unet', 'willnet', 'willnet_se', or 'willnet_se_plus'")
    parser.add_argument("--gamma", type=float, default=0.5, help="Weight ratio of MSE to SSIM in loss function")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--mid_channels", type=int, default=48, help="Middle channels for WillNetSEPlus")
    parser.add_argument("--n_blocks", type=int, default=8, help="Number of residual blocks for WillNetSEPlus")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        print(f"Warning: Config file {config_path} is empty or invalid. Using default values.")
        config = {}
    
    return config


def create_model(model_name, device, args=None):
    """
    Create a model instance based on model name.
    
    Args:
        model_name (str): Name of the model ('unet', 'willnet', or 'willnet_se')
        device (torch.device): Device to move the model to
        
    Returns:
        nn.Module: Model instance
    """
    if model_name.lower() == 'unet':
        model = SRNET().to(device)
    elif model_name.lower() == 'willnet':
        model = WillNet().to(device)
    elif model_name.lower() == 'willnet_se':  # Add support for new model
        model = WillNetSE().to(device)
    elif model_name.lower() == 'willnet_se_plus':
        mid_channels = args.mid_channels if args else 48
        n_blocks = args.n_blocks if args else 8
        model = WillNetSEPlus(
            in_channels=1, 
            out_channels=1, 
            mid_channels=mid_channels,
            n_blocks=n_blocks,
            upscale=2
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'unet', 'willnet', or 'willnet_se'")
    
    return model


def train_epoch(model, dataloader, optimizer, gamma, device, model_name):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        gamma (float): Weight for MSE loss in combined loss
        device (torch.device): Device to use for training
        model_name (str): Name of the model for handling different input requirements
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for i, (lr_img, hr_img) in enumerate(tqdm(dataloader, desc="Training")):
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)
        
        optimizer.zero_grad()
        
        if model_name.lower() == 'willnet_se_plus':
            sr_img = model(lr_img)
        else:
            lr_img_up = torch.nn.functional.interpolate(
                lr_img, 
                scale_factor=2, 
                mode='bicubic', 
                align_corners=False
            )
            sr_img = model(lr_img_up)
        
        loss = combined_loss(sr_img, hr_img, gamma)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, gamma, device, model_name):
    """
    Validate the model on the validation set.
    
    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        gamma (float): Weight for MSE loss in combined loss
        device (torch.device): Device to use for validation
        
    Returns:
        tuple: (average validation loss, metrics dictionary)
    """
    model.eval()
    total_loss = 0
    total_metrics = {'psnr': 0.0, 'ssim': 0.0}
    
    with torch.no_grad():
        for lr_img, hr_img in tqdm(dataloader, desc="Validation"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            if model_name.lower() == 'willnet_se_plus':
                sr_img = model(lr_img)
            else:
                lr_img_up = torch.nn.functional.interpolate(
                    lr_img,
                    scale_factor=2,
                    mode='bicubic',
                    align_corners=False
                )
                sr_img = model(lr_img_up)
            
            loss = combined_loss(sr_img, hr_img, gamma)
            total_loss += loss.item()
            
            batch_metrics = evaluate_metrics(sr_img, hr_img)
            for k, v in batch_metrics.items():
                total_metrics[k] += v
    
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


def save_checkpoint(model, optimizer, epoch, loss, metrics, output_dir, model_name):
    """
    Save a model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (torch.optim.Optimizer): Optimizer state
        epoch (int): Current epoch
        loss (float): Validation loss
        metrics (dict): Validation metrics
        output_dir (str): Directory to save checkpoint
        model_name (str): Name of the model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    
    checkpoint_path = os.path.join(output_dir, f"{model_name}_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main training function."""
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir / f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"))
    
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = Path(config.get("data_dir", "./data"))
    transforms_config = config.get("transforms_config", "./configs/transforms.yaml")
    
    train_loader, val_loader = create_dataloaders(
        data_dir=str(data_dir),
        config_path=transforms_config,
        loader_to_create="both",
        batch_size=args.batch_size,
        num_workers=config.get("num_workers", 4)
    )
    
    model = create_model(args.model, device, args)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, args.gamma, device, args.model)
        train_losses.append(train_loss)
        
        val_loss, val_metrics = validate(model, val_loader, args.gamma, device, args.model)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        print(f"Val PSNR: {val_metrics['psnr']:.2f} dB, Val SSIM: {val_metrics['ssim']:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/psnr', val_metrics['psnr'], epoch)
        writer.add_scalar('Metrics/ssim', val_metrics['ssim'], epoch)
        
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_metrics, 
            str(output_dir), args.model
        )
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(str(output_dir / f"{args.model}_loss_plot.png"))
    
    print(f"Training completed. Model saved to {output_dir}")


if __name__ == "__main__":
    main()