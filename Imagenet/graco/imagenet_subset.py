"""ImageNet-100 subset helpers for Inception experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from torchvision import transforms


def default_imagenet_root() -> Path | None:
    r = os.environ.get("IMAGENET_ROOT", "").strip()
    if not r:
        return None
    return Path(r).expanduser().resolve()


def alibi_rgb_transform() -> transforms.Compose:
    """RGB: Resize 299, ToTensor [0,1], shift to Alibi [-0.5, 0.5] per channel."""
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t - 0.5).float()),
        ]
    )


def load_imagenet_val_manifest_samples(
    root: str | Path,
    manifest_entries: list[dict],
    *,
    max_samples: int | None = None,
    class_map_path: Path | None = None,
) -> List[Tuple[int, int, np.ndarray]]:
    """Val images for eval_manifest.json entries (same indices as generate_cfproto etc.)."""
    root_p = Path(root).expanduser().resolve() if root is not None else default_imagenet_root()
    if root_p is None:
        raise ValueError(
            "ImageNet root not set. Set IMAGENET_ROOT or pass root= for manifest loading."
        )
    if not root_p.is_dir():
        raise FileNotFoundError(f"IMAGENET_ROOT is not a directory: {root_p}")

    import sys

    _imnet = Path(__file__).resolve().parents[1]
    if str(_imnet) not in sys.path:
        sys.path.insert(0, str(_imnet))
    from imagenet100_dataset import ImageNet100AlibiDataset
    from imagenet_data import resolve_class_map_path

    entries = list(manifest_entries)
    if max_samples is not None and max_samples > 0:
        entries = entries[: int(max_samples)]

    cmap = resolve_class_map_path(class_map_path)
    tform = alibi_rgb_transform()

    out: List[Tuple[int, int, np.ndarray]] = []
    for rec in entries:
        idx = int(rec.get("dataset_index", rec.get("sample_idx", -1)))
        if idx < 0:
            raise ValueError(f"Manifest entry missing dataset_index / sample_idx: {rec!r}")
        y_exp = int(rec["y_true"])
        split = str(rec.get("split", "val")).lower()
        ds = ImageNet100AlibiDataset(root_p, split, class_map_path=cmap, transform=tform)  # type: ignore[arg-type]
        img_t, y_ds = ds[idx]
        if int(y_ds) != y_exp:
            raise ValueError(
                f"Manifest y_true={y_exp} does not match dataset label {int(y_ds)} at index {idx}"
            )
        arr = np.ascontiguousarray(img_t.numpy(), dtype=np.float32)
        if arr.shape != (3, 299, 299):
            raise RuntimeError(f"Unexpected image shape {arr.shape} at index {idx}, expected (3, 299, 299)")
        out.append((idx, y_exp, arr))
    return out
