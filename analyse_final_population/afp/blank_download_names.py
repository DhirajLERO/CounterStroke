"""Download filename stems for blank-canvas exports."""

from __future__ import annotations

import re

# Short tags for change-map method ids in download names.
MAP_METHOD_STEM: dict[str, str] = {
    "sum_intensity": "sum_intensity",
    "mean_intensity": "mean_intensity",
    "change_frequency": "change_frequency",
    "max_intensity": "max_intensity",
    "cochange_strict": "cochange",
    "spatial_coherence": "spatial_coherence",
}


def dataset_stem(dataset: str) -> str:
    """Normalize dataset name for filenames (mnist / cifar10 / imagenet100)."""
    return str(dataset).strip().lower().replace(" ", "")


def class_label_stem(dataset: str, class_idx: int, class_names: list[str] | None = None) -> str:
    """MNIST keeps digit indices; CIFAR/ImageNet use sanitized class names."""
    ds = dataset_stem(dataset)
    if ds == "mnist" or not class_names:
        return str(int(class_idx))
    if 0 <= int(class_idx) < len(class_names):
        raw = str(class_names[int(class_idx)])
    else:
        raw = str(int(class_idx))
    # Filename-safe: lowercase, spaces/punct → underscore
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw.strip().lower()).strip("_")
    return slug or str(int(class_idx))


def blank_cf_stem(
    dataset: str,
    class_idx: int,
    pred_prob: float,
    class_names: list[str] | None = None,
) -> str:
    """e.g. mnist_0_90_42 or cifar10_dog_90_42 for p=0.9042."""
    hundredths = int(round(float(pred_prob) * 10000))
    hundredths = max(0, min(hundredths, 10000))
    whole, frac = divmod(hundredths, 100)
    cls = class_label_stem(dataset, class_idx, class_names)
    return f"{dataset_stem(dataset)}_{cls}_{whole}_{frac:02d}"


def blank_ssim_stem(
    dataset: str,
    class_idx: int,
    pred_prob: float,
    class_names: list[str] | None = None,
) -> str:
    return f"{blank_cf_stem(dataset, class_idx, pred_prob, class_names)}_ssim"


def blank_map_stem(
    dataset: str,
    class_idx: int,
    method_id: str,
    class_names: list[str] | None = None,
) -> str:
    """e.g. mnist_0_cochange or cifar10_dog_cochange."""
    tag = MAP_METHOD_STEM.get(method_id, method_id.replace("-", "_"))
    cls = class_label_stem(dataset, class_idx, class_names)
    return f"{dataset_stem(dataset)}_{cls}_{tag}"
