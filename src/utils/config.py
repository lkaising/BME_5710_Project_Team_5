"""Unified YAML‑driven config loader for MRI SR pipeline."""
from __future__ import annotations
import yaml
from pathlib import Path
import sys
from typing import Dict, Any

from typing import Any, Mapping, Union
from types import SimpleNamespace

try:
    from rich.console import Console
    from rich.table import Table

    _USE_RICH = True
    _console = Console()
except ImportError:               # graceful fallback
    _USE_RICH = False

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

# def display_args(args, title="Training arguments:", indent=2, padding=12):
#     print(title)
#     items = vars(args) if hasattr(args, "__dict__") else args
#     prefix = " " * indent
#     for key in sorted(items):
#         print(f"{prefix}{key:<{padding}} = {items[key]!r}")

def _to_mapping(obj: Union[Mapping[str, Any], SimpleNamespace, object]) -> Mapping[str, Any]:
    """Return a read-only dict-like view of namespace or plain object."""
    if isinstance(obj, Mapping):
        return obj
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise TypeError(f"Unsupported type for display_args: {type(obj).__name__}")


def display_args(
    args: Union[Mapping[str, Any], SimpleNamespace, object],
    *,
    title: str = "Training arguments",
    indent: int = 2,
    padding: int = 14,
    stream = sys.stdout,
) -> None:
    """
    Nicely print a mapping / argparse.Namespace / SimpleNamespace of arguments.

    Parameters
    ----------
    args : Mapping | Namespace | object
        Anything with a ``__dict__`` attribute or a mapping interface.
    title : str
        Heading printed above the key-value pairs.
    indent : int
        Number of leading spaces before each line (plain-text mode).
    padding : int
        Minimum width reserved for the key column (plain-text mode).
    stream : IO
        Where to write output when *rich* is unavailable (defaults to stdout).
    """
    items = _to_mapping(args)

    if _USE_RICH:
        table = Table(title=title, show_header=False, pad_edge=False, box=None)
        table.add_column(justify="left")
        table.add_column(justify="left")

        for k in sorted(items):
            table.add_row(f"[bold]{k}[/bold]", repr(items[k]))
        _console.print(table)
    else:
        stream.write(f"{title}\n")
        pref = " " * indent
        for k in sorted(items):
            stream.write(f"{pref}{k:<{padding}} = {items[k]!r}\n")