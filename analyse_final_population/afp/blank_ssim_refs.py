"""Cached val/test pools and SSIM nearest-neighbour for blank-canvas UI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch

from .graco_shared import load_ssim, load_val_by_class

# Interactive SSIM on ImageNet: cap pool size per class.
IMAGENET_SSIM_POOL_CAP = 200


@st.cache_resource(show_spinner="Loading class image pool…")
def cached_val_by_class(
    dataset: str,
    data_root: str,
    class_map: str | None = None,
) -> dict[int, list[np.ndarray]]:
    """Load display-[0,1] val/test images grouped by class (cached)."""
    vbc = load_val_by_class()
    root = Path(data_root)
    if dataset == "mnist":
        return vbc.val_pool_mnist(root)
    if dataset == "cifar10":
        return vbc.val_pool_cifar(root)
    if dataset == "imagenet100":
        cmap = Path(class_map) if class_map else None
        by = vbc.val_pool_imagenet(root, cmap)
        # Cap for interactive SSIM latency.
        return {k: imgs[:IMAGENET_SSIM_POOL_CAP] for k, imgs in by.items()}
    raise ValueError(f"Unknown dataset {dataset!r}")


def best_ssim_ref(
    cf_01: np.ndarray,
    pool_01: list[np.ndarray],
    *,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray | None, float]:
    """Return (best reference image in [0,1], SSIM) or (None, nan)."""
    if not pool_01:
        return None, float("nan")
    ssim_mod = load_ssim()
    dev = torch.device(device) if isinstance(device, str) else device
    idx, score = ssim_mod.best_ssim_match(cf_01, pool_01, device=dev)
    if idx < 0:
        return None, float("nan")
    return pool_01[idx], float(score)
