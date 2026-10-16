"""Render phenotypes to Alibi images via dataset Graco turtle modules."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .datasets import AppDatasetSpec
from .population_io import IndividualRecord


@lru_cache(maxsize=4)
def _load_hooks(dataset_name: str, graco_dir: str):
    from .graco_shared import load_graco_blank_evolve

    mod = load_graco_blank_evolve()
    return mod.load_turtle_hooks(Path(graco_dir), dataset=dataset_name)


def render_phenotype_alibi(
    spec: AppDatasetSpec,
    phenotype: str,
    base_alibi: np.ndarray,
) -> np.ndarray:
    from .graco_shared import load_graco_mine

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
    spec: AppDatasetSpec,
    record: IndividualRecord,
    base_alibi: np.ndarray,
) -> np.ndarray:
    if record.image_alibi is not None:
        return np.asarray(record.image_alibi, dtype=np.float32)
    return render_phenotype_alibi(spec, record.phenotype, base_alibi)
