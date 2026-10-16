"""Aggregate blank-canvas final-generation metrics per target class."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .analysis import PROB_THRESHOLD, filter_flipped_high_conf, pred_class_distribution
from .blank_paths import iter_target_classes, load_target_population
from pathlib import Path

from .datasets import AppDatasetSpec
from .population_io import IndividualRecord


@dataclass(frozen=True)
class TargetBeliefRow:
    wanted_class: int
    n_total: int
    success_rate: float
    mean_p_target: float
    mode_pred: int | None
    mean_p_blank: float
    n_flip_hi: int
    has_csv: bool


@dataclass
class TargetDrilldown:
    wanted_class: int
    n_total: int
    success_rate: float
    mean_p_blank: float
    mode_pred: int | None
    dist_full: dict[int, int]
    dist_filtered: dict[int, int]
    records: list[IndividualRecord]
    filtered: list[IndividualRecord]
    o1_values: list[float]


def _success_metrics(records: list[IndividualRecord], k: int) -> tuple[float, float]:
    hits = [r for r in records if r.pred_class == k and r.pred_prob >= PROB_THRESHOLD]
    if not records:
        return 0.0, 0.0
    success_rate = len(hits) / len(records)
    mean_p = float(np.mean([r.pred_prob for r in hits])) if hits else 0.0
    return success_rate, mean_p


def build_belief_landscape(
    spec: AppDatasetSpec,
    graco_run_root: Path,
    *,
    run_id: int = 0,
) -> list[TargetBeliefRow]:
    rows: list[TargetBeliefRow] = []
    targets = iter_target_classes(graco_run_root)
    if not targets:
        targets = list(range(spec.num_classes))

    for k in targets:
        recs = load_target_population(graco_run_root, k, run_id)
        has_csv = bool(recs)
        if not recs:
            rows.append(
                TargetBeliefRow(
                    wanted_class=k,
                    n_total=0,
                    success_rate=0.0,
                    mean_p_target=0.0,
                    mode_pred=None,
                    mean_p_blank=0.0,
                    n_flip_hi=0,
                    has_csv=False,
                )
            )
            continue

        success_rate, mean_p_target = _success_metrics(recs, k)
        dist = pred_class_distribution(recs)
        mode_pred = dist.most_common(1)[0][0] if dist else None
        p_blank_vals = [r.p_original_class for r in recs if r.p_original_class is not None]
        mean_p_blank = float(np.mean(p_blank_vals)) if p_blank_vals else 0.0
        filtered = filter_flipped_high_conf(recs)

        rows.append(
            TargetBeliefRow(
                wanted_class=k,
                n_total=len(recs),
                success_rate=success_rate,
                mean_p_target=mean_p_target,
                mode_pred=mode_pred,
                mean_p_blank=mean_p_blank,
                n_flip_hi=len(filtered),
                has_csv=True,
            )
        )
    return rows


def load_target_drilldown(
    graco_run_root: Path,
    wanted_class: int,
    *,
    run_id: int = 0,
) -> TargetDrilldown:
    recs = load_target_population(graco_run_root, wanted_class, run_id)
    filtered = filter_flipped_high_conf(recs)
    success_rate, _ = _success_metrics(recs, wanted_class)
    p_blank_vals = [r.p_original_class for r in recs if r.p_original_class is not None]
    mean_p_blank = float(np.mean(p_blank_vals)) if p_blank_vals else 0.0
    dist_full = dict(pred_class_distribution(recs))
    dist_filtered = dict(pred_class_distribution(filtered))
    mode_pred = (
        Counter(r.pred_class for r in recs).most_common(1)[0][0] if recs else None
    )
    o1_vals = [float(r.o1) for r in recs if r.o1 is not None]

    return TargetDrilldown(
        wanted_class=wanted_class,
        n_total=len(recs),
        success_rate=success_rate,
        mean_p_blank=mean_p_blank,
        mode_pred=mode_pred,
        dist_full=dist_full,
        dist_filtered=dist_filtered,
        records=recs,
        filtered=filtered,
        o1_values=o1_vals,
    )
