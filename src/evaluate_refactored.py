#!/usr/bin/env python3
# evaluate_refactored.py – Consistent evaluation entry-point for MRI super-resolution
# -----------------------------------------------------------------------------
# Highlights
#   • Evaluator class encapsulates dataloader creation, metric loop, logging.
#   • YAML config + CLI merge for reproducibility (same style as train script).
#   • Automatic model-builder supports all architectures the trainer knows.
#   • Metrics computed for bicubic-interpolated baseline **and** SR output.
#   • JSON summary + optional PNG comparison saved to --output_dir.
#   • Torch AMP enabled on CUDA for a small speed boost (no GradScaler needed).
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast
from tqdm import tqdm

from models import (
    WillNet,
    WillNetSEDeep,
    combined_loss,  # not used here but keeps IDE import grouping consistent
)
from utils import create_dataloaders, evaluate_metrics
from utils.visualization import plot_comparison  # <- supplies a neat 3-pane image


# --------------------------------------------------------------------------
# CLI ----------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate MRI super-resolution model")
    p.add_argument("--config", type=str, required=True, help="YAML config file")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (*.pth)")
    p.add_argument("--output_dir", type=str, default="./results", help="Where to save metrics & figures")
    p.add_argument("--model", type=str, default="willnet", choices=[
        "unet", "willnet", "willnet_se", "willnet_se_plus", "willnet_se_deep"
    ], help="Model architecture")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    p.add_argument("--mid_channels", type=int, default=48, help="Width for SEPlus / SEDeep (when needed)")
    p.add_argument("--n_blocks", type=int, default=8, help="#ResBlocks for SEDeep")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--save_visual", action="store_true", help="Save LR/SR/HR PNG of first batch")
    return p


# --------------------------------------------------------------------------
# Helpers ------------------------------------------------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_model(name: str, device: torch.device, args: argparse.Namespace) -> torch.nn.Module:
    name = name.lower()
    if name == "willnet":
        model = WillNet()
    elif name == "willnet_se_deep":
        model = WillNetSEDeep(n_blocks=args.n_blocks, mid_channels=args.mid_channels)
    else:
        raise ValueError(f"Unknown model {name}")
    return model.to(device)


# --------------------------------------------------------------------------
# Evaluator ----------------------------------------------------------------

class Evaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # I/O ----------------------------------------------------------------
        self.out_dir = Path(args.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_stamp = time.strftime("%Y%m%d_%H%M%S")

        # Data ---------------------------------------------------------------
        cfg = load_yaml(args.config)
        data_dir = Path(cfg.get("data_dir", "./data"))
        transforms_cfg = cfg.get("transforms_config", "./configs/transforms.yaml")
        self.loader = create_dataloaders(
            data_dir=str(data_dir),
            config_path=transforms_cfg,
            loader_to_create="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Model --------------------------------------------------------------
        self.model = build_model(args.model, self.device, args)
        self._load_checkpoint(args.checkpoint)
        self.model.eval()

    # ----------------------------------------------------------------------
    def _load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        print(f"Model weights loaded from {path}")

        # ckpt = torch.load(path, map_location=self.device)
        # state_dict = ckpt["model_state_dict"]
        
        # new_state_dict = {}
        # for key, value in state_dict.items():
        #     if key.startswith("blocks."):
        #         new_key = key.replace("blocks.", "body.")
        #         new_state_dict[new_key] = value
        #     else:
        #         new_state_dict[key] = value
        
        # self.model.load_state_dict(new_state_dict)
        # print(f"Model weights loaded from {path} (with key renaming)")

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def run(self) -> Dict[str, Dict[str, float]]:
        tot_interp, tot_sr = {"psnr": 0.0, "ssim": 0.0}, {"psnr": 0.0, "ssim": 0.0}

        iterator = tqdm(self.loader, desc="Evaluating", leave=False)
        for b_idx, (lr_img, hr_img) in enumerate(iterator):
            lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)

            with autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                lr_up = F.interpolate(lr_img, scale_factor=2, mode="bicubic", align_corners=False)
                sr_img = self.model(lr_up if self.args.model != "willnet_se_plus" else lr_img)

            # Metrics -----------------------------------------------------
            m_interp = evaluate_metrics(lr_up, hr_img)
            m_sr = evaluate_metrics(sr_img, hr_img)
            for k in tot_interp:
                tot_interp[k] += m_interp[k]
                tot_sr[k] += m_sr[k]

            # Save a single illustrative comparison ----------------------
            if self.args.save_visual and b_idx == 0:
                self._save_visuals(lr_up, sr_img, hr_img)

        # Averages ----------------------------------------------------------
        n = len(self.loader)
        avg_interp = {k: v / n for k, v in tot_interp.items()}
        avg_sr = {k: v / n for k, v in tot_sr.items()}
        improvement = {k: avg_sr[k] - avg_interp[k] for k in avg_sr}

        results = {"interpolated": avg_interp, "super_resolved": avg_sr, "improvement": improvement}
        results_path = self.out_dir / f"evaluation_{self.run_stamp}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\nEvaluation complete ✔  Metrics saved to {results_path}")
        self._pretty_print(results)
        return results

    # ----------------------------------------------------------------------
    def _save_visuals(self, lr_up: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor):
        """Save LR↑, SR, HR and error maps for the first item in batch 0."""
        lr_np = lr_up[0].squeeze().cpu().numpy()
        sr_np = sr[0].squeeze().cpu().numpy()
        hr_np = hr[0].squeeze().cpu().numpy()

        # Comparison PNG -------------------------------------------------
        cmp_path = self.out_dir / f"comparison_{self.run_stamp}.png"
        plot_comparison(lr_np, sr_np, hr_np, save_path=str(cmp_path))

        # NPY dumps ------------------------------------------------------
        np.save(self.out_dir / "lr_up.npy", lr_np)
        np.save(self.out_dir / "sr.npy", sr_np)
        np.save(self.out_dir / "hr.npy", hr_np)
        np.save(self.out_dir / "err_interp.npy", np.abs(hr_np - lr_np) * 5)
        np.save(self.out_dir / "err_sr.npy", np.abs(hr_np - sr_np) * 5)
        print(f"Saved visual sample to {cmp_path}")

        import csv
        
        def save_as_csv(arr, name):
            csv_path = self.out_dir / f"{name}_{self.run_stamp}.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for row in arr:
                    writer.writerow(row)
        
        save_as_csv(lr_np, "lr_up_matrix")
        save_as_csv(sr_np, "sr_matrix")
        save_as_csv(hr_np, "hr_matrix")
        save_as_csv(np.abs(hr_np - lr_np) * 5, "err_interp_matrix")
        save_as_csv(np.abs(hr_np - sr_np) * 5, "err_sr_matrix")

    # ----------------------------------------------------------------------
    @staticmethod
    def _pretty_print(res: Dict[str, Dict[str, float]]):
        interp, sr, imp = res["interpolated"], res["super_resolved"], res["improvement"]
        print(
            f"\n{'Metric':<15}{'Bicubic (↑)':>15}{'Super-Res':>15}{'ΔSR-Bicubic':>15}"
            f"\n{'-'*60}"
        )
        for k in ["psnr", "ssim"]:
            print(f"{k.upper():<15}{interp[k]:>15.2f}{sr[k]:>15.2f}{imp[k]:>15.2f}")


# --------------------------------------------------------------------------
# Entry-point ---------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()
    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
