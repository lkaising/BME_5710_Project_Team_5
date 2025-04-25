"""Unified YAML‑driven config loader for MRI SR pipeline."""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any

class Config(dict):
    """Dict-like config object with attribute access (+merge)."""

    def __getattr__(self, k):  # dot access
        return self[k] if k in self else None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge(a: Dict, b: Dict) -> Dict:
        out = a.copy()
        for k, v in b.items():
            if isinstance(v, dict) and k in out:
                out[k] = Config._merge(out[k], v)
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path, base: str | Path = "configs/base_config.yaml") -> "Config":
        with open(base, "r") as f:
            base_cfg = yaml.safe_load(f) or {}
        with open(path, "r") as f:
            exp_cfg = yaml.safe_load(f) or {}
        merged = cls._merge(base_cfg, exp_cfg)
        return cls(merged)

    # convenience ------------------------------------------------------ #
    def to_loader_kwargs(self):
        d = self["data"]
        return dict(
            data_dir=d["dir"],
            config_path=self["paths"]["transforms_yaml"],
            loader_to_create="both",
            batch_size=self["training"]["batch_size"],
            num_workers=d.get("num_workers", 4),
        )
    


# from pathlib import Path, PurePath
# import yaml, copy, types

# def _merge(parent: dict, child: dict) -> dict:
#     merged = copy.deepcopy(parent)
#     merged.update(child)
#     return merged

# def load_cfg(path: str | Path) -> types.MappingProxyType:
#     path = Path(path)
#     cfg = yaml.safe_load(path.read_text()) or {}
#     parent_path = cfg.pop("parent", None)
#     if parent_path:
#         parent_cfg = load_cfg(path.parent / parent_path)
#         cfg = _merge(parent_cfg, cfg)
#     return types.MappingProxyType(cfg)
