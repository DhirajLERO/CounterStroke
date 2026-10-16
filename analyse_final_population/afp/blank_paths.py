"""Discover blank-canvas analysis CSVs under graco_run/target_XXX/."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .population_io import IndividualRecord, rows_from_csv

_TARGET_RE = re.compile(r"^target_(\d+)$")


def load_summary_blank_pred(outputs_dir: Path) -> int | None:
    summary = outputs_dir.parent / "summary.json"
    if not summary.is_file():
        return None
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
        return int(data["blank_pred"])
    except (KeyError, TypeError, ValueError):
        return None


def target_csv_path(graco_run_root: Path, wanted_class: int, run_id: int = 0) -> Path:
    return (
        graco_run_root
        / f"target_{wanted_class:03d}"
        / f"run_{run_id}"
        / f"run_{run_id}_final_generation.csv"
    )


def iter_target_classes(graco_run_root: Path) -> list[int]:
    if not graco_run_root.is_dir():
        return []
    out: list[int] = []
    for p in sorted(graco_run_root.iterdir()):
        m = _TARGET_RE.match(p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return out


def load_target_population(
    graco_run_root: Path,
    wanted_class: int,
    run_id: int = 0,
) -> list[IndividualRecord]:
    csv_p = target_csv_path(graco_run_root, wanted_class, run_id)
    if not csv_p.is_file():
        return []
    return rows_from_csv(csv_p)
