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

from typing import Mapping, Sequence
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import shutil  
from torch.amp import autocast
from tqdm import tqdm
from tabulate import tabulate

from models import (
    TrivialNet,
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
    p.add_argument("--mid_channels", type=int, default=64, help="Width for SEPlus / SEDeep (when needed)")
    p.add_argument("--n_blocks", type=int, default=10, help="#ResBlocks for SEDeep")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--save_visual", action="store_true", help="Save LR/SR/HR PNG of first batch")
    p.add_argument("--split",      type=str, default="test", choices=["train", "val", "test"], help="Which dataset partition to evaluate on")
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
        # print(f"Using device: {self.device}")

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
            loader_to_create=args.split,
            batch_size=args.batch_size,
            num_workers=cfg.get("num_workers", 4),
        )

        # Model --------------------------------------------------------------
        self.model = build_model(args.model, self.device, args)
        self._load_checkpoint(args.checkpoint)
        self.model.eval()
        # ------- learned-baseline (TrivialNet) ---------------------------
        self.baseline = TrivialNet().to(self.device)
        triv_ckpt = Path("checkpoints/trivialnet/best.pth")
        if triv_ckpt.exists():
            self.baseline.load_state_dict(torch.load(triv_ckpt, map_location=self.device)["model"])
            self.baseline.eval()
        else:
            raise FileNotFoundError(
                "TrivialNet checkpoint not found"
            )

    # ----------------------------------------------------------------------
    def _load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        # print(f"Model weights loaded from {path}")


    # ----------------------------------------------------------------------
    @torch.no_grad()
    def run(self) -> Dict[str, Dict[str, float]]:
        # totals ----------------------------------------------------------------
        tot_bic  = {"psnr": 0.0, "ssim": 0.0}
        tot_triv = {"psnr": 0.0, "ssim": 0.0}
        tot_will = {"psnr": 0.0, "ssim": 0.0}

        iterator = tqdm(self.loader, desc="Evaluating", leave=False)
        for b_idx, (lr_img, hr_img) in enumerate(iterator):
            lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)

            # forward --------------------------------------------------------
            with autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                lr_up     = F.interpolate(lr_img, scale_factor=2, mode="bicubic",
                                        align_corners=False)          # bicubic ↑
                triv_img  = self.baseline(lr_up)                         # TrivialNet
                will_img  = self.model(lr_up if self.args.model != "willnet_se_plus"
                                    else lr_img)                      # WillNet

            # metrics -------------------------------------------------------
            m_bic  = evaluate_metrics(lr_up , hr_img)
            m_triv = evaluate_metrics(triv_img, hr_img)
            m_will = evaluate_metrics(will_img, hr_img)
            for k in tot_bic:
                tot_bic [k] += m_bic [k]
                tot_triv[k] += m_triv[k]
                tot_will[k] += m_will[k]

            # one illustrative figure --------------------------------------
            if self.args.save_visual and b_idx == 0:
                self._save_visuals(lr_up, will_img, triv_img, hr_img)

        # averages -------------------------------------------------------------
        n           = len(self.loader)
        avg_bic     = {k: v / n for k, v in tot_bic .items()}
        avg_triv    = {k: v / n for k, v in tot_triv.items()}
        avg_will    = {k: v / n for k, v in tot_will.items()}
        imp_vs_bic  = {k: avg_will[k] - avg_bic [k] for k in avg_will}
        imp_vs_triv = {k: avg_will[k] - avg_triv[k] for k in avg_will}

        results = {
            "bicubic"       : avg_bic,
            "trivialnet"    : avg_triv,
            "willnet"       : avg_will,
            "imp_vs_bicubic": imp_vs_bic,
            "imp_vs_trivial": imp_vs_triv,
        }

        results_path = self.out_dir / f"evaluation_{self.run_stamp}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

        self._pretty_print(results)
        return results

    # ----------------------------------------------------------------------
    def _save_visuals(
        self,
        lr_up   : torch.Tensor,
        sr_will : torch.Tensor,
        sr_triv : torch.Tensor,
        hr      : torch.Tensor
    ):
        """
        Save LR↑, WillNet-SR, TrivialNet-SR, HR and their error maps for the
        first item in batch 0.
        """
        # tensors → numpy ---------------------------------------------------
        lr_np   = lr_up  [0].squeeze().cpu().numpy()
        will_np = sr_will[0].squeeze().cpu().numpy()
        triv_np = sr_triv[0].squeeze().cpu().numpy()
        hr_np   = hr     [0].squeeze().cpu().numpy()

        # figure ------------------------------------------------------------
        cmp_path = self.out_dir / f"comparison_{self.run_stamp}.png"
        plot_comparison(lr_np, will_np, triv_np, hr_np, save_path=str(cmp_path))

        # numpy dumps -------------------------------------------------------
        np.save(self.out_dir / "lr_up.npy"      , lr_np  )
        np.save(self.out_dir / "sr_will.npy"    , will_np)
        np.save(self.out_dir / "sr_triv.npy"    , triv_np)
        np.save(self.out_dir / "hr.npy"         , hr_np  )
        np.save(self.out_dir / "err_bic.npy"    , np.abs(hr_np - lr_np  ) * 5)
        np.save(self.out_dir / "err_will.npy"   , np.abs(hr_np - will_np) * 5)
        np.save(self.out_dir / "err_triv.npy"   , np.abs(hr_np - triv_np) * 5)

        # CSV versions ------------------------------------------------------
        import csv, itertools
        def save_csv(arr, name):
            path = self.out_dir / f"{name}_{self.run_stamp}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(arr)

        save_csv(lr_np , "lr_up_matrix")
        save_csv(will_np, "sr_will_matrix")
        save_csv(triv_np, "sr_triv_matrix")
        save_csv(hr_np , "hr_matrix")
        save_csv(np.abs(hr_np - lr_np  ) * 5, "err_bic_matrix")
        save_csv(np.abs(hr_np - will_np) * 5, "err_will_matrix")
        save_csv(np.abs(hr_np - triv_np) * 5, "err_triv_matrix")

    # ----------------------------------------------------------------------
    @staticmethod
    def _pretty_print(res: Dict[str, Dict[str, float]]) -> None:
        """
        Console table showing Bicubic, TrivialNet, and WillNet metrics plus
        WillNet’s improvements over both baselines.
        """
        from tabulate import tabulate

        metrics = ["psnr", "ssim"]
        headers = [
            "Metric",
            "Bicubic (↑)",
            "TrivialNet (↑)",
            "WillNet (↑)",
            "ΔWill-Bic",
            "ΔWill-Triv",
        ]

        rows = []
        for m in metrics:
            bic   = res["bicubic"][m]
            triv  = res["trivialnet"][m]
            will  = res["willnet"][m]
            rows.append([
                m.upper(),
                f"{bic:.4f}",
                f"{triv:.4f}",
                f"{will:.4f}",
                f"{will - bic:+.4f}",
                f"{will - triv:+.4f}",
            ])

        print(tabulate(rows, headers=headers, tablefmt="simple"))


# --------------------------------------------------------------------------
# Entry-point ---------------------------------------------------------------

def main():
    args = build_arg_parser().parse_args()
    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
