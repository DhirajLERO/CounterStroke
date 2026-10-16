"""tensors.npz schema for counterfactual outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

METHOD_KEYS = [
    ("counterfactual_pt", "cf_counterfactual_pt"),
    ("cfproto", "cf_cfproto"),
    ("cfrl", "cf_cfrl"),
    ("leapfactual", "cf_leapfactual"),
    ("leapfactual_r", "cf_leapfactual_r"),
    ("graco_base_o2", "cf_graco_base_o2"),
    ("graco_base_o3", "cf_graco_base_o3"),
    ("graco50", "cf_graco50"),
]


def invalid_cf_tensor(shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
    return np.full(shape, np.nan, dtype=dtype)


def save_sample_npz(
    path: Path,
    *,
    orig_nhwc: np.ndarray,
    cfs: dict[str, np.ndarray],
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, np.ndarray] = {"orig_nhwc": np.asarray(orig_nhwc, dtype=np.float32)}
    for k, arr in cfs.items():
        payload[k] = np.asarray(arr, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: np.asarray(z[k]) for k in z.files}
