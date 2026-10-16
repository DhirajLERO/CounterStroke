"""Render phenotypes and load original factual images."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import DatasetSpec
from ..data.graco_paths import summary_npz
from ..data.population import IndividualRecord
from ..imports_helper import load_graco_blank_evolve, load_graco_mine


@lru_cache(maxsize=4)
def _load_hooks(dataset_name: str, graco_dir: str):
    mod = load_graco_blank_evolve()
    return mod.load_turtle_hooks(Path(graco_dir), dataset=dataset_name)


def original_image_01_to_alibi(spec: DatasetSpec, img_01: np.ndarray) -> np.ndarray:
    img = np.asarray(img_01, dtype=np.float32)
    if spec.name == "imagenet100":
        if img.ndim == 3 and img.shape[0] == 3:
            return img - 0.5
        return np.transpose(img, (2, 0, 1)) - 0.5
    return img - 0.5


def load_original_alibi(run_path: Path, spec: DatasetSpec, *, run_id: int = 0) -> np.ndarray | None:
    npz_path = summary_npz(run_path, run_id)
    if not npz_path.is_file():
        return None
    data = np.load(npz_path, allow_pickle=True)
    if "original_image" not in data:
        return None
    return original_image_01_to_alibi(spec, data["original_image"])


def render_phenotype_alibi(
    spec: DatasetSpec,
    phenotype: str,
    base_alibi: np.ndarray,
) -> np.ndarray:
    render_cf_01 = load_graco_mine().render_cf_01
    hooks = _load_hooks(spec.name, str(spec.graco_dir.resolve()))
    cf_01 = render_cf_01(
        phenotype,
        base_alibi,
        hooks,
        graco_dir=spec.graco_dir,
        turtle_delta_scale=spec.turtle_delta_scale,
    )
    if spec.name == "imagenet100":
        return np.transpose((cf_01.astype(np.float32) - 0.5), (2, 0, 1))
    return (cf_01.astype(np.float32) - 0.5).astype(np.float32)


def record_to_alibi(
    spec: DatasetSpec,
    record: IndividualRecord,
    base_alibi: np.ndarray,
) -> np.ndarray:
    if record.image_alibi is not None:
        return np.asarray(record.image_alibi, dtype=np.float32)
    return render_phenotype_alibi(spec, record.phenotype, base_alibi)
