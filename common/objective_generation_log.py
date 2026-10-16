"""Per-generation O1/O2/O3 logging and run-id scheduling for CounterStroke turtles."""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

OBJECTIVE_COLUMNS = (
    "avg_o1",
    "avg_o2",
    "avg_o3",
    "min_o1",
    "min_o2",
    "min_o3",
)

GENERATION_LOG_COLUMNS = (
    "gen",
    "invalid",
    "avg",
    "std",
    "min",
    "max",
    "fitness_test",
    "best_ind_length",
    "avg_length",
    "best_ind_nodes",
    "avg_nodes",
    "best_ind_depth",
    "avg_depth",
    "avg_used_codons",
    "best_ind_used_codons",
    "structural_diversity",
    "selection_time",
    "generation_time",
    *OBJECTIVE_COLUMNS,
)


def evolution_seed(cfg: dict, loop_index: int) -> int:
    """Seed 10 is the finished run_0. run_id 1 and 2 are seeds 11 and 12."""
    return 10 + int(cfg.get("run_id", loop_index))


def scheduled_run_ids(parallel_cfg: dict, runs_per_sample: int) -> list[int]:
    """Prefer an explicit run_ids list so run_0 is not scheduled again."""
    raw = parallel_cfg.get("run_ids")
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        return [int(x) for x in raw]
    start = int(parallel_cfg.get("run_id_start", 0))
    return list(range(start, start + int(runs_per_sample)))


def _obj_reduce(fits: Any, reducer) -> np.ndarray:
    arr = np.asarray(list(fits), dtype=np.float64)
    if arr.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.asarray(reducer(arr, axis=0), dtype=np.float64).reshape(-1)
    if out.size < 3:
        out = np.pad(out, (0, 3 - int(out.size)), constant_values=np.nan)
    return out[:3]


def register_objective_stats(stats) -> None:
    """Keep the old scalar avg/min, and add per-objective mean and best."""
    stats.register("avg_obj", lambda fits: _obj_reduce(fits, np.nanmean))
    stats.register("min_obj", lambda fits: _obj_reduce(fits, np.nanmin))


def _triple(row: Any) -> list[float]:
    arr = np.asarray(row, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - int(arr.size)), constant_values=np.nan)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def objective_row_suffix(avg_obj: Any, min_obj: Any) -> list[float]:
    return _triple(avg_obj) + _triple(min_obj)


def write_objectives_by_generation(results_dir: str, logbook) -> str:
    """Small CSV the collector reads. One row per generation, objectives split."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "objectives_by_generation.csv")
    gens = list(logbook.select("gen"))
    avgs = list(logbook.select("avg_obj"))
    mins = list(logbook.select("min_obj"))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gen", *OBJECTIVE_COLUMNS])
        for i, gen in enumerate(gens):
            writer.writerow([int(gen), *objective_row_suffix(avgs[i], mins[i])])
    return path
