"""ImageNet-100 ImageFolder dataset with Alibi [-0.5, 0.5] tensors and class id 0..99."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from graco.imagenet_subset import alibi_rgb_transform
from imagenet100_classmap import load_class_map, resolve_class_map_path

Split = Literal["train", "val"]


class ImageNet100AlibiDataset(Dataset):
    """__getitem__ returns (x_nchw, class_id) with x in Alibi [-0.5, 0.5]."""

    def __init__(
        self,
        root: Path | str,
        split: Split,
        *,
        class_map_path: Path | None = None,
        transform=None,
    ) -> None:
        self.root = Path(root).resolve()
        self.split: Split = split
        self._cmap = load_class_map(resolve_class_map_path(class_map_path))
        split_dir = self.root / split
        if transform is None:
            transform = alibi_rgb_transform()
        self._inner = ImageFolder(str(split_dir), transform=transform)
        expected_wnids = self._cmap["wnids"]
        for i, wnid in enumerate(self._inner.classes):
            if wnid != expected_wnids[i]:
                raise ValueError(
                    f"Class order mismatch at {i}: folder={wnid!r} map={expected_wnids[i]!r}"
                )

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img, class_id = self._inner[index]
        return img, int(class_id)

    @property
    def targets(self) -> list[int]:
        return [int(y) for y in self._inner.targets]

    @property
    def wnids(self) -> list[str]:
        return list(self._inner.classes)

    @property
    def class_names(self) -> list[str]:
        return list(self._cmap["class_names"])


def subset_loader(
    ds: ImageNet100AlibiDataset,
    indices: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=shuffle and len(indices) > batch_size,
    )


def gather_nhwc_batch(
    ds: ImageNet100AlibiDataset,
    indices: list[int],
    *,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load indices into NHWC float32 Alibi arrays (for small batches / eval)."""
    _ = device
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for i in indices:
        img, y = ds[int(i)]
        arr = img.numpy().astype(np.float32)
        xs.append(np.transpose(arr, (1, 2, 0)))
        ys.append(int(y))
    x_np = np.stack(xs, axis=0)
    y_np = np.asarray(ys, dtype=np.int64)
    return x_np, y_np


def load_train_pool_indices(path: Path) -> list[int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [int(x) for x in data["indices"]]


def manifest_entry_split(entry: dict) -> Split:
    sp = str(entry.get("split", "val")).lower()
    if sp not in ("train", "val"):
        raise ValueError(f"Unknown manifest split {sp!r}")
    return sp  # type: ignore[return-value]


def manifest_dataset_index(entry: dict) -> int:
    if "dataset_index" in entry:
        return int(entry["dataset_index"])
    return int(entry["sample_idx"])
