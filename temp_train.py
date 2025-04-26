#!/usr/bin/env python3
"""
Quick-and-dirty training script for TrivialNet.
It writes   checkpoints/trivialnet/best.pth
            (and last.pth for resume purposes)
"""
import argparse, os, sys, torch, torch.nn as nn, torch.optim as optim, yaml
from pathlib import Path

PROJECT_ROOT: Path = Path.cwd()
SRC_DIR:       Path = PROJECT_ROOT / "src"
CONFIG_DIR:    Path = PROJECT_ROOT / "configs"

sys.path.insert(0, str(SRC_DIR))

from utils          import create_dataloaders          # already in your repo
from models      import TrivialNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       required=True)         # YAML – same as before
    ap.add_argument("--epochs",       type=int, default=100)
    ap.add_argument("--batch_size",   type=int, default=5)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--save_dir",     type=str,   default="checkpoints/trivialnet")
    args = ap.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg        = yaml.safe_load(Path(args.config).read_text())
    train_ld   = create_dataloaders(
                    data_dir      = cfg["data"]["dir"],
                    config_path   = cfg["data"]["transforms"],
                    loader_to_create="train",          # always train split
                    batch_size    = args.batch_size,
                    num_workers   = cfg.get("num_workers", 4),
                 )

    model      = TrivialNet().to(device)
    optimiser  = optim.Adam(model.parameters(), lr=args.lr)
    criterion  = nn.MSELoss()

    best_loss  = float("inf")
    save_dir   = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train(); running = 0.0
        for lr_im, hr_im in train_ld:
            lr_im, hr_im = lr_im.to(device), hr_im.to(device)
            lr_im = torch.nn.functional.interpolate(lr_im, scale_factor=2,
                                                    mode="bicubic", align_corners=False)
            optimiser.zero_grad()
            loss = criterion(model(lr_im), hr_im)
            loss.backward(); optimiser.step()
            running += loss.item()

        avg = running/len(train_ld)
        print(f"[{epoch:03d}/{args.epochs}] train-loss={avg:.6f}")

        # save last & best
        torch.save({"epoch":epoch, "model":model.state_dict()},
                   save_dir/"last.pth")
        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch":epoch, "model":model.state_dict()},
                       save_dir/"best.pth")

if __name__ == "__main__":
    main()
