"""Helpers for Alibi NHWC tensors → RGB display images and matplotlib panels."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_SAMPLE_DIR_RE = re.compile(r"sample_(\d+)$", re.IGNORECASE)


def parse_sample_dir_idx(sample_dir: Path) -> int:
    """Parse trailing digits from ``sample_XXXXX`` folder names (any width)."""
    m = _SAMPLE_DIR_RE.match(sample_dir.name)
    if not m:
        raise ValueError(f"Not a sample_* folder name: {sample_dir.name}")
    return int(m.group(1))


def alibi_nhwc_to_display_rgb(x: np.ndarray) -> np.ndarray:
    """Map Alibi [-0.5, 0.5] NHWC to float RGB [0, 1], shape (H, W, 3)."""
    t = np.asarray(x, dtype=np.float64)
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(f"Expected batch 1, got shape {t.shape}")
        t = t[0]
    if t.ndim != 3:
        raise ValueError(f"Expected NHWC after squeeze, got {t.shape}")
    c = t.shape[-1]
    if c == 1:
        g = np.clip(t[..., 0] + 0.5, 0.0, 1.0)
        return np.stack([g, g, g], axis=-1).astype(np.float32)
    if c == 3:
        return np.clip(t + 0.5, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Expected 1 or 3 channels, got C={c}")


def panel_valid(arr: np.ndarray | None) -> bool:
    if arr is None:
        return False
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return False
    if not np.isfinite(a).any():
        return False
    if np.isnan(a).all():
        return False
    return True


def tensor_to_panel_hwc(
    arr: np.ndarray | None,
    *,
    ref_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, bool]:
    h_def, w_def = (ref_hw if ref_hw is not None else (32, 32))
    gray = np.full((h_def, w_def, 3), 0.55, dtype=np.float32)
    if not panel_valid(arr):
        return gray, False
    try:
        rgb = alibi_nhwc_to_display_rgb(np.asarray(arr, dtype=np.float32))
    except (ValueError, FloatingPointError):
        return gray, False
    if rgb.shape[0] < 1 or rgb.shape[1] < 1:
        return gray, False
    return rgb, True


def load_tensors_npz(npz_path: Path) -> dict[str, np.ndarray]:
    z = np.load(npz_path)
    return {k: np.asarray(z[k]) for k in z.files}


def reference_hw_from_orig(orig: np.ndarray | None) -> tuple[int, int]:
    if orig is None or not panel_valid(orig):
        return 32, 32
    t = np.asarray(orig, dtype=np.float32)
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3:
        return 32, 32
    return int(t.shape[0]), int(t.shape[1])


# (npz_key, subplot_title)
METHOD_COLUMNS: list[tuple[str, str]] = [
    ("orig_nhwc", "Original"),
    ("cf_cfproto", "CFProto"),
    ("cf_cfrl", "CFRL"),
    ("cf_counterfactual_pt", "Grad CF"),
    ("cf_leapfactual", "LeapFactual"),
    ("cf_leapfactual_r", "LeapFactual-R"),
    ("cf_graco_base_o2", "Graco base_o2"),
    ("cf_graco_base_o3", "Graco base_o3"),
    ("cf_graco50", "Graco50"),
]
