"""Import one dataset turtle and load the same images the main run used."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from typing import Any

import numpy as np

from spec import SPECS, DatasetSpec, manifest_sample_idx

_LOADED: dict[str, ModuleType] = {}


def import_turtle(dataset: str) -> ModuleType:
    """Load the turtle with its grammar directory as the working directory."""
    if dataset in _LOADED:
        return _LOADED[dataset]
    if _LOADED:
        loaded = next(iter(_LOADED))
        raise RuntimeError(
            f"Turtle for {loaded!r} is already imported in this process. "
            "Run each dataset in its own process."
        )
    spec = SPECS[dataset]
    if not spec.turtle_py.is_file():
        raise FileNotFoundError(spec.turtle_py)
    graco = str(spec.graco_dir)
    if graco in sys.path:
        sys.path.remove(graco)
    sys.path.insert(0, graco)
    os.chdir(spec.graco_dir)
    module_name = f"_eval_budget_turtle_{dataset}"
    module_spec = importlib.util.spec_from_file_location(module_name, spec.turtle_py)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Cannot load {spec.turtle_py}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    _LOADED[dataset] = module
    return module


def y_intended_of(entry: dict[str, Any], y_true: int, dataset: str) -> int:
    if entry.get("y_intended_target") is not None:
        return int(entry["y_intended_target"])
    n_classes = 100 if dataset == "imagenet" else 10
    return (int(y_true) + 1) % n_classes


def load_image(
    dataset: str,
    turtle: ModuleType,
    entry: dict[str, Any],
    *,
    imagenet_root: str | None,
) -> tuple[int, int, np.ndarray, int]:
    """Return ``(sample_idx, y_true, image, y_intended)`` in the turtle's layout."""
    spec = SPECS[dataset]
    if dataset == "imagenet":
        if not imagenet_root:
            raise SystemExit("ImageNet needs --imagenet-root or IMAGENET_ROOT.")
        rows = turtle.load_imagenet_val_manifest_samples(imagenet_root, [entry], max_samples=1)
        if not rows:
            raise RuntimeError(f"ImageNet manifest entry did not load: {entry!r}")
        sample_idx, y_true, image = rows[0]
        image = np.ascontiguousarray(image, dtype=np.float32)
        return int(sample_idx), int(y_true), image, y_intended_of(entry, int(y_true), dataset)

    if not spec.data_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {dataset} data directory {spec.data_dir}. "
            "The budget run does not download datasets."
        )
    raw = turtle.load_dataset(data_dir=str(spec.data_dir), download=False)
    sample_idx = manifest_sample_idx(entry)
    img_t, y_true = raw[int(entry.get("sample_idx", sample_idx))]
    image = np.ascontiguousarray(img_t.numpy(), dtype=np.float32)
    y_true = int(y_true)
    expected = int(entry["y_true"])
    if y_true != expected:
        raise RuntimeError(
            f"{dataset} sample {sample_idx}: manifest y_true={expected} but the dataset label is {y_true}."
        )
    return sample_idx, y_true, image, y_intended_of(entry, y_true, dataset)


def to_nhwc_alibi(dataset: str, image: np.ndarray) -> np.ndarray:
    """Evolution layout -> ``(1, H, W, C)`` Alibi, which is what the metric scripts read."""
    spec: DatasetSpec = SPECS[dataset]
    arr = np.asarray(image, dtype=np.float32)
    if dataset == "imagenet":
        hwc = np.transpose(arr, (1, 2, 0))
        return hwc[np.newaxis, ...].astype(np.float32, copy=False)
    if dataset == "mnist":
        hw = arr.reshape(28, 28)
        return hw[np.newaxis, :, :, np.newaxis].astype(np.float32, copy=False)
    hwc = arr.reshape(spec.spatial_shape)
    return hwc[np.newaxis, ...].astype(np.float32, copy=False)


def render_phenotype(
    dataset: str,
    turtle: ModuleType,
    image: np.ndarray,
    phenotype: str,
    *,
    turtle_delta_scale: float,
) -> np.ndarray:
    scale = float(turtle_delta_scale)
    if dataset == "imagenet":
        batch = turtle.process_phenotypes_for_images_rgb(
            [phenotype], image, turtle_delta_scale=scale
        )
        return to_nhwc_alibi(dataset, batch[0])
    batch = turtle.process_phenotypes_for_images(
        [phenotype], image, turtle_delta_scale=scale
    )
    return to_nhwc_alibi(dataset, batch[0])


def invalid_cf(dataset: str) -> np.ndarray:
    return np.full((1, *SPECS[dataset].spatial_shape), np.nan, dtype=np.float32)
