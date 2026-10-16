"""Resolve true label when pickle entries omit original_class (MNIST final_pop.pkl)."""

from __future__ import annotations

from pathlib import Path

from .graco_paths import summary_npz
from .population import IndividualRecord
import numpy as np


def resolve_original_class(
    run_path: Path,
    records: list[IndividualRecord],
    y_true: int,
    *,
    run_id: int = 0,
) -> int:
    """Pick the factual true class: record field, summary npz, then manifest y_true."""
    for rec in records:
        if rec.original_class >= 0:
            return int(rec.original_class)

    npz_path = summary_npz(run_path, run_id)
    if npz_path.is_file():
        data = np.load(npz_path, allow_pickle=True)
        if "original_class" in data:
            return int(data["original_class"])

    return int(y_true)


def normalize_record_original_classes(records: list[IndividualRecord], original_class: int) -> None:
    """Patch pickle rows that default to -1 so flip filtering uses the real label."""
    oc = int(original_class)
    for rec in records:
        if rec.original_class < 0:
            rec.original_class = oc
