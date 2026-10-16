"""Load final-generation individuals (pickle preferred, else CSV)."""

from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .graco_paths import final_generation_csv, final_population_pickle, run_dir


@dataclass
class IndividualRecord:
    phenotype: str
    pred_class: int
    pred_prob: float
    original_class: int
    p_original_class: float | None
    o1: float | None
    o2: float | None
    o3: float | None
    image_alibi: np.ndarray | None


def rows_from_csv(csv_path: Path) -> list[IndividualRecord]:
    rows: list[IndividualRecord] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            p_orig = rec.get("p_original_class")
            rows.append(
                IndividualRecord(
                    phenotype=str(rec["phenotype"]),
                    pred_class=int(rec["pred_class"]),
                    pred_prob=float(rec["pred_prob"]),
                    original_class=int(rec["original_class"]),
                    p_original_class=float(p_orig) if p_orig is not None else None,
                    o1=float(rec["o1"]) if "o1" in rec else None,
                    o2=float(rec["o2"]) if "o2" in rec else None,
                    o3=float(rec["o3"]) if "o3" in rec else None,
                    image_alibi=None,
                )
            )
    return rows


def _rows_from_pickle(pkl_path: Path) -> list[IndividualRecord]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    rows: list[IndividualRecord] = []
    for entry in data:
        img = entry.get("image")
        rows.append(
            IndividualRecord(
                phenotype=str(entry.get("phenotype", "")),
                pred_class=int(entry["pred_class"]),
                pred_prob=float(entry["pred_prob"]),
                original_class=int(entry.get("original_class", entry.get("label_true", -1))),
                p_original_class=float(entry["p_true_class"])
                if "p_true_class" in entry
                else None,
                o1=float(entry["o1_score"]) if "o1_score" in entry else None,
                o2=float(entry["sparsity_pixels"]) if "sparsity_pixels" in entry else None,
                o3=float(entry["proximity_l2"]) if "proximity_l2" in entry else None,
                image_alibi=np.asarray(img, dtype=np.float32) if img is not None else None,
            )
        )
    return rows


def load_final_population(
    sample_dir: Path,
    *,
    run_id: int = 0,
) -> tuple[list[IndividualRecord], str]:
    """Return records and source label ('pkl' | 'csv')."""
    rd = run_dir(sample_dir, run_id)
    pkl = final_population_pickle(rd, run_id)
    if pkl.is_file():
        return _rows_from_pickle(pkl), "pkl"
    csv_p = final_generation_csv(rd, run_id)
    if csv_p.is_file():
        return rows_from_csv(csv_p), "csv"
    return [], "none"
