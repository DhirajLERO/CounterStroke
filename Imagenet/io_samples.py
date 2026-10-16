"""Load manifest + merge npz (same as MNIST)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_npz(path: Path, updates: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, np.ndarray] = {}
    if path.is_file():
        z = np.load(path)
        for k in z.files:
            data[k] = np.asarray(z[k])
    data.update({k: np.asarray(v, dtype=np.float32) for k, v in updates.items()})
    np.savez_compressed(path, **data)


def load_or_create_meta(sample_dir: Path, entry: dict[str, Any]) -> dict[str, Any]:
    p = sample_dir / "meta.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    sid = int(entry.get("sample_idx", entry.get("dataset_index", 0)))
    meta = {
        "sample_idx": sid,
        "dataset_index": int(entry.get("dataset_index", sid)),
        "y_true": int(entry["y_true"]),
        "y_intended_target": int(entry["y_intended_target"]),
        "y_target_shift": int(entry["y_intended_target"]),
        "methods": {},
    }
    return meta


def save_meta(sample_dir: Path, meta: dict[str, Any]) -> None:
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def sample_dir_name(idx: int) -> str:
    return f"sample_{idx:08d}"
