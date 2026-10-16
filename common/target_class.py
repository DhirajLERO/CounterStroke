"""Per-class confusion CSV + target class resolution."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ClassConfusionRow:
    class_id: int
    class_name: str
    top1_accuracy: float
    top_confuser_id: Optional[int]
    top_confuser_count: int
    top_confuser_rate: float


def load_per_class_confusion_csv(path: Path) -> list[ClassConfusionRow]:
    rows: list[ClassConfusionRow] = []
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rec in r:
            cid = int(rec["class_id"])
            name = str(rec.get("class_name", ""))
            acc = float(rec["top1_accuracy"])
            tc_raw = rec.get("top_confuser_id", "").strip()
            tc_id: Optional[int]
            if tc_raw == "" or tc_raw.upper() in ("NA", "NAN", "NONE"):
                tc_id = None
            else:
                tc_id = int(tc_raw)
            tc_cnt = int(rec.get("top_confuser_count", 0))
            tc_rate = float(rec.get("top_confuser_rate", 0.0))
            rows.append(
                ClassConfusionRow(
                    class_id=cid,
                    class_name=name,
                    top1_accuracy=acc,
                    top_confuser_id=tc_id,
                    top_confuser_count=tc_cnt,
                    top_confuser_rate=tc_rate,
                )
            )
    return rows


def classes_best_accuracy_first(rows: list[ClassConfusionRow], n: int) -> list[int]:
    """Highest top1_accuracy first (least confused / 'bottom' confusion)."""
    sorted_rows = sorted(rows, key=lambda x: x.top1_accuracy, reverse=True)
    return [r.class_id for r in sorted_rows[:n]]


def target_class_for_row(row: ClassConfusionRow, probs: np.ndarray) -> int:
    """Target for CFProto/CFRL/Leap: top_confuser_id from CSV if valid and != class_id;"""
    y_true = row.class_id
    p = np.asarray(probs, dtype=np.float64).ravel()
    if row.top_confuser_id is not None and int(row.top_confuser_id) != y_true:
        return int(row.top_confuser_id)
    mask = np.ones_like(p, dtype=bool)
    mask[y_true] = False
    idx = int(np.argmax(p * mask))
    return idx


def row_for_class(rows: list[ClassConfusionRow], class_id: int) -> ClassConfusionRow | None:
    for r in rows:
        if r.class_id == class_id:
            return r
    return None
