# train_refactored.py – Cleaner training entry‑point for MRI super‑resolution
# ---------------------------------------------------------------------------
# Highlights
#   • Trainer class isolates loop logic (train/val, checkpoint, logging).
#   • Mixed‑precision (AMP) + GradScaler for free speed on GPU.
#   • StepLR scheduler, best‑val checkpoint, optional early stopping.
#   • YAML config merged with CLI for reproducibility.
#   • Supports SRNET, WillNet, WillNetSE, WillNetSEPlus.
# ---------------------------------------------------------------------------
from __future__ import annotations

import os, gc, argparse, time 
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import WillNet, WillNetSEDeep, combined_loss
from utils import create_dataloaders, evaluate_metrics

# ---------------------------------------------------------------------------
# CLI -----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MRI super-resolution model")
    p.add_argument("--config", type=str, required=True, help="YAML config file")
    p.add_argument("--checkpoint", type=str, help="Resume checkpoint path")
    p.add_argument("--output_dir", type=str, default="./checkpoints", help="Save dir")
    p.add_argument("--model", type=str, default="willnet", choices = [
        "willnet", "willnet_se_deep"], help="Model arch")
    p.add_argument("--gamma", type=float, default=0.5, help="MSE:SSIM weight ratio")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--batch_size", type=int, default=4, help="Batch size")
    p.add_argument("--epochs", type=int, default=100, help="Max epochs")
    p.add_argument("--mid_channels", type=int, default=48, help="Width for SEPlus")
    p.add_argument("--n_blocks", type=int, default=8, help="#ResBlocks for SEPlus")
    p.add_argument("--patience", type=int, default=20, help="Early-stop patience")
    p.add_argument("--step_size", type=int, default=30, help="LR StepLR period")
    p.add_argument("--gamma_lr", type=float, default=0.5, help="LR decay factor")
    return p

# ---------------------------------------------------------------------------
# Utilities -----------------------------------------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def build_model(name: str, device: torch.device, args: argparse.Namespace) -> nn.Module:
    name = name.lower()
    if name == "willnet":
        model = WillNet()
    elif name == "willnet_se_deep":
        model = WillNetSEDeep(n_blocks=args.n_blocks, mid_channels=args.mid_channels)
    else:
        raise ValueError(f"Unknown model {name}")
    return model.to(device)

# ---------------------------------------------------------------------------
# Trainer -------------------------------------------------------------------

class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # I/O ----------------------------------------------------------------
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path("./logs") / f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=str(log_dir))

        # Data ---------------------------------------------------------------
        cfg = load_yaml(args.config)
        data_dir = Path(cfg.get("data_dir", "./data"))
        transforms_cfg = cfg.get("transforms_config", "./configs/transforms.yaml")
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir=str(data_dir),
            config_path=transforms_cfg,
            loader_to_create="both",
            batch_size=args.batch_size,
            num_workers=cfg.get("num_workers", 4),
        )

        # Model/optim --------------------------------------------------------
        self.model = build_model(args.model, self.device, args)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        # self.scheduler = optim.lr_scheduler.StepLR(
        #     self.optimizer, step_size=args.step_size, gamma=args.gamma_lr
        # )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )
        self.scaler = GradScaler(enabled=self.device.type == "cuda")

        # State --------------------------------------------------------------
        self.best_loss = float("inf")
        self.start_epoch = 0
        if args.checkpoint:
            self._load_checkpoint(args.checkpoint)

    # ---------------------------------------------------------------------
    # Epoch routines -------------------------------------------------------
    def _forward_pass(self, lr_img: torch.Tensor, model_name: str) -> torch.Tensor:
        lr_up = torch.nn.functional.interpolate(lr_img, scale_factor=2, mode="bicubic", align_corners=False)
        return self.model(lr_up)

    def _loop(self, loader, train: bool) -> Tuple[float, Dict[str, float]]:
        mode = "train" if train else "val"
        self.model.train(mode == "train")
        total_loss, total_metrics = 0.0, {"psnr": 0.0, "ssim": 0.0}

        iterator = tqdm(loader, desc=mode.capitalize(), leave=False)
        for lr_img, hr_img in iterator:
            lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)

            with autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                sr_img = self._forward_pass(lr_img, self.args.model.lower())
                loss = combined_loss(sr_img, hr_img, self.args.gamma)

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            if not train:
                batch_metrics = evaluate_metrics(sr_img, hr_img)
                for k, v in batch_metrics.items():
                    total_metrics[k] += v

        avg_loss = total_loss / len(loader)
        if not train:
            total_metrics = {k: v / len(loader) for k, v in total_metrics.items()}
        return avg_loss, total_metrics

    # ---------------------------------------------------------------------
    # Checkpoint -----------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]):
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": val_loss,
            "metrics": metrics,
        }
        fname = self.output_dir / f"{self.args.model}_epoch_{epoch}_test.pth"
        torch.save(ckpt, fname)
        print(f"Checkpoint saved: {fname}")

    def _load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_loss = ckpt.get("loss", float("inf"))
        print(f"Resumed from {path} @ epoch {self.start_epoch}")

    # ---------------------------------------------------------------------
    # Public API -----------------------------------------------------------
    def fit(self):
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")

            if epoch % 10 == 0: torch.cuda.empty_cache(); gc.collect()

            train_loss, _ = self._loop(self.train_loader, train=True)
            val_loss, val_metrics = self._loop(self.val_loader, train=False)

            self.scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Logging -----------------------------------------------------
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalar("Metrics/psnr", val_metrics["psnr"], epoch)
            self.writer.add_scalar("Metrics/ssim", val_metrics["ssim"], epoch)
            print(f"Loss ➜ train: {train_loss:.4f} | val: {val_loss:.4f} | PSNR: {val_metrics['psnr']:.2f} | SSIM: {val_metrics['ssim']:.4f}")

            # Checkpointing ----------------------------------------------
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(epoch, val_loss, val_metrics)
            else:
                patience_counter += 1

            if patience_counter >= self.args.patience:
                print("Early stopping triggered.")
                break

        self._plot_losses(train_losses, val_losses)
        print(f"Training complete. Best val loss: {self.best_loss:.4f}")

    # ---------------------------------------------------------------------
    def _plot_losses(self, train: list[float], val: list[float]):
        plt.figure(figsize=(10, 5))
        plt.plot(train, label="Train")
        plt.plot(val, label="Validation")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        fname = self.output_dir / f"{self.args.model}_loss_curve.png"
        plt.savefig(fname)
        print(f"Loss curve saved: {fname}")

# ---------------------------------------------------------------------------
# Entry‑point ---------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()
    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()