"""Load manifest samples from ImageNet-100."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from imagenet100_dataset import (
    ImageNet100AlibiDataset,
    manifest_dataset_index,
    manifest_entry_split,
)
from imagenet_data import resolve_class_map_path


def load_chw_alibi(
    root: Path,
    split: str,
    dataset_index: int,
    *,
    class_map_path: Path | None = None,
) -> tuple[np.ndarray, int]:
    ds = ImageNet100AlibiDataset(
        root,
        split,  # type: ignore[arg-type]
        class_map_path=resolve_class_map_path(class_map_path),
    )
    img_t, y = ds[int(dataset_index)]
    arr = np.ascontiguousarray(img_t.numpy(), dtype=np.float32)
    return arr, int(y)


def load_manifest_entry_nhwc(
    entry: dict,
    root: Path,
    *,
    class_map_path: Path | None = None,
) -> tuple[np.ndarray, int, int]:
    """Return ``(image_nhwc, y_true class_id 0..99, dataset_index)``."""
    split = manifest_entry_split(entry)
    idx = manifest_dataset_index(entry)
    chw, y = load_chw_alibi(root, split, idx, class_map_path=class_map_path)
    nhwc = np.transpose(chw, (1, 2, 0))
    y_true = int(entry.get("y_true", y))
    return nhwc, y_true, idx


def load_manifest_batch_nhwc(
    entries: list[dict],
    root: Path,
    *,
    class_map_path: Path | None = None,
) -> np.ndarray:
    """Load manifest images only; labels and intended targets remain manifest data."""
    xs: list[np.ndarray] = []
    for e in entries:
        nhwc, _y_true, _ = load_manifest_entry_nhwc(e, root, class_map_path=class_map_path)
        xs.append(nhwc)
    return np.stack(xs, axis=0)
