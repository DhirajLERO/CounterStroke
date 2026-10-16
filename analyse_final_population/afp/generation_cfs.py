"""Load per-sample comparison CFs from ``outputs/generation/sample_*/tensors.npz``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .datasets import AppDatasetSpec
from .metrics_catalog import METHOD_ORDER

METHOD_NPZ_KEY = {m: f"cf_{m}" for m in METHOD_ORDER}

METHOD_SHORT_LABEL = {
    "counterfactual_pt": "Wachter",
    "cfproto": "CFProto",
    "cfrl": "CFRL",
    "leapfactual": "Leap",
    "leapfactual_r": "Leap-R",
    "graco_base_o2": "Graco O2",
    "graco_base_o3": "Graco O3",
    "graco50": "Graco50",
}

GRACO_METHODS = ("graco_base_o2", "graco_base_o3", "graco50")
BASELINE_METHODS = tuple(m for m in METHOD_ORDER if m not in GRACO_METHODS)


def sample_dir_name(dataset: str, sample_idx: int) -> str:
    if dataset == "imagenet100":
        return f"sample_{int(sample_idx):08d}"
    return f"sample_{int(sample_idx):05d}"


def generation_sample_dir(spec: AppDatasetSpec, sample_idx: int, gen_root: Path | None = None) -> Path:
    root = Path(gen_root) if gen_root is not None else spec.default_gen_root
    return root / sample_dir_name(spec.name, sample_idx)


def _squeeze_nhwc(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    return a


def nhwc_alibi_to_display_01(arr: np.ndarray) -> np.ndarray | None:
    """Alibi NHWC [-0.5, 0.5] → display [0, 1] HWC (gray or RGB). Returns None if invalid."""
    a = _squeeze_nhwc(arr)
    if a.size == 0 or not np.isfinite(a).any() or np.isnan(a).all():
        return None
    if a.ndim != 3:
        return None
    c = a.shape[-1]
    if c == 1:
        return np.clip(a[..., 0] + 0.5, 0.0, 1.0)
    if c == 3:
        return np.clip(a + 0.5, 0.0, 1.0)
    return None


def load_sample_meta(sample_dir: Path) -> dict:
    p = sample_dir / "meta.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_method_cfs(
    spec: AppDatasetSpec,
    sample_idx: int,
    *,
    gen_root: Path | None = None,
    methods: tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray | None]:
    """Return method → display-[0,1] image (or None if missing/invalid)."""
    sample_dir = generation_sample_dir(spec, sample_idx, gen_root=gen_root)
    npz_path = sample_dir / "tensors.npz"
    wanted = list(methods) if methods is not None else list(METHOD_ORDER)
    out: dict[str, np.ndarray | None] = {m: None for m in wanted}
    if not npz_path.is_file():
        return out
    try:
        z = np.load(npz_path)
    except Exception:
        return out
    for method in wanted:
        key = METHOD_NPZ_KEY.get(method, f"cf_{method}")
        if key not in z.files:
            continue
        out[method] = nhwc_alibi_to_display_01(z[key])
    return out


def method_caption(method: str, meta: dict) -> str:
    short = METHOD_SHORT_LABEL.get(method, method)
    gsr = meta.get("graco_selected_row") or {}
    if method in gsr and isinstance(gsr[method], dict):
        ent = gsr[method]
        p = ent.get("pred_prob")
        pc = ent.get("pred_class")
        if p is not None and pc is not None:
            return f"{short} · pred={pc} p={float(p):.2f}"
    methods_meta = meta.get("methods") or {}
    if method in methods_meta and isinstance(methods_meta[method], dict):
        ent = methods_meta[method]
        status = ent.get("status") or ent.get("ok")
        if status is not None:
            return f"{short} · {status}"
    return short
