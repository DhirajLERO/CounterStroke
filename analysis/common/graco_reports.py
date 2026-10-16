"""CSV helpers for blank-canvas Graco runs (vendored from Imagenet/graco/io_reports.py)."""

from __future__ import annotations

import csv

import numpy as np


def write_final_generation_csv(
    *,
    out_csv_path: str,
    phenotypes: list[str],
    o1: np.ndarray,
    o2: np.ndarray,
    o3: np.ndarray,
    original_class: int,
    p_original: np.ndarray,
    pred_class: np.ndarray,
    pred_prob: np.ndarray,
) -> None:
    header = [
        "ind_index",
        "phenotype",
        "o1",
        "o2",
        "o3",
        "original_class",
        "p_original_class",
        "pred_class",
        "pred_prob",
    ]
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, ph in enumerate(phenotypes):
            w.writerow(
                [
                    i,
                    ph,
                    float(o1[i]),
                    float(o2[i]),
                    float(o3[i]),
                    int(original_class),
                    float(p_original[i]),
                    int(pred_class[i]),
                    float(pred_prob[i]),
                ]
            )
