"""ImageNet-100 paths and dataset factory."""

from __future__ import annotations

import argparse
from pathlib import Path

from imagenet100_classmap import default_class_map_path, resolve_class_map_path
from imagenet100_dataset import ImageNet100AlibiDataset, Split

__all__ = [
    "Split",
    "ImageNet100AlibiDataset",
    "add_imagenet_root_args",
    "default_class_map_path",
    "resolve_class_map_path",
    "get_alibi_dataset",
]


def add_imagenet_root_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--imagenet-root",
        type=Path,
        required=True,
        help="ImageNet-100 root with train/ and val/ ImageFolder splits.",
    )
    ap.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="imagenet100_class_map.json (default: Imagenet/data_subset/...)",
    )


def get_alibi_dataset(
    root: Path,
    split: Split,
    *,
    class_map_path: Path | None = None,
) -> ImageNet100AlibiDataset:
    return ImageNet100AlibiDataset(
        root,
        split,
        class_map_path=resolve_class_map_path(class_map_path),
    )
