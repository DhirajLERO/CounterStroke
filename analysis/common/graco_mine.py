"""Pick Graco strategies from final-generation CSV and render counterfactuals."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .graco_blank_evolve import TurtleHooks


@dataclass(frozen=True)
class Row:
    o1: float
    o2: float
    o3: float
    original_class: int
    pred_class: int
    pred_prob: float
    phenotype: str


def read_final_generation_csv(csv_path: Path) -> list[Row]:
    rows: list[Row] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            rows.append(
                Row(
                    o1=float(rec["o1"]),
                    o2=float(rec["o2"]),
                    o3=float(rec["o3"]),
                    original_class=int(rec["original_class"]),
                    pred_class=int(rec["pred_class"]),
                    pred_prob=float(rec["pred_prob"]),
                    phenotype=str(rec["phenotype"]),
                )
            )
    return rows


def _pick_min_o2_then_o3(cands: list[Row]) -> Row | None:
    if not cands:
        return None
    return min(cands, key=lambda r: (r.o2, r.o3))


def _pick_min_o3_then_o2(cands: list[Row]) -> Row | None:
    if not cands:
        return None
    return min(cands, key=lambda r: (r.o3, r.o2))


def _pick_graco_threshold(cands: list[Row], threshold: float) -> Row | None:
    hi = [r for r in cands if float(r.pred_prob) >= float(threshold)]
    return _pick_min_o2_then_o3(hi)


def pick_strategies(
    rows: list[Row],
    wanted_class: int,
    *,
    require_flip: bool = True,
) -> dict[str, Row | None]:
    cands = [r for r in rows if r.pred_class == int(wanted_class)]
    if require_flip:
        cands = [r for r in cands if r.pred_class != r.original_class]
    return {
        "graco_base_o2": _pick_min_o2_then_o3(cands),
        "graco_base_o3": _pick_min_o3_then_o2(cands),
        "graco50": _pick_graco_threshold(cands, 0.50),
        "graco60": _pick_graco_threshold(cands, 0.60),
        "graco70": _pick_graco_threshold(cands, 0.70),
        "graco80": _pick_graco_threshold(cands, 0.80),
        "graco90": _pick_graco_threshold(cands, 0.90),
    }


def render_cf_01(
    phenotype: str,
    blank_image: np.ndarray,
    hooks: TurtleHooks,
    *,
    graco_dir: Path,
    turtle_delta_scale: float,
) -> np.ndarray:
    old = os.getcwd()
    os.chdir(graco_dir)
    try:
        batch = hooks.process_phenotypes(
            [phenotype], blank_image, turtle_delta_scale=turtle_delta_scale
        )
        out = batch[0]
        if out.ndim == 3 and out.shape[0] == 3:
            from apply_turtle_rgb import alibi_chw_to_hwc_01  # noqa: WPS433

            return alibi_chw_to_hwc_01(out)
        return hooks.to_display_01(out)
    finally:
        os.chdir(old)
