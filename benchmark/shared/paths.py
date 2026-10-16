"""Path helpers for benchmark experiments (vendor package + dataset roots)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
GRACOIMAGE_ROOT = BENCHMARK_ROOT.parent
VENDOR_ROOT = BENCHMARK_ROOT / "vendor"
VENDOR_PKG = VENDOR_ROOT / "counterfactual_benchmark"
DEEPSCM_DIR = VENDOR_PKG / "methods" / "deepscm"


def setup_vendor_path() -> None:
    """Ensure vendored counterfactual_benchmark is importable."""
    for p in (str(VENDOR_ROOT), str(VENDOR_PKG), str(DEEPSCM_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


def deepscm_cwd(dataset_dir: Path) -> None:
    """Run train/eval with cwd = dataset folder (configs use relative paths)."""
    os.chdir(dataset_dir)


def dataset_root(name: str) -> Path:
    if name == "morphomnist":
        return BENCHMARK_ROOT / "morphomnist"
    if name == "celeba":
        return BENCHMARK_ROOT / "celeba"
    raise ValueError(f"Unknown dataset: {name}")


def morphomnist_data_dir() -> Path:
    return dataset_root("morphomnist") / "data"


def celeba_data_dir() -> Path:
    return dataset_root("celeba") / "data"
