"""Resolve per-sample Graco result directories."""

from __future__ import annotations

from pathlib import Path


def find_graco_sample_dir(graco_results_root: Path, sample_idx: int, y_true: int) -> Path | None:
    exact = graco_results_root / f"{sample_idx}_label_{y_true}"
    if exact.is_dir():
        return exact
    matches = sorted(graco_results_root.glob(f"{sample_idx}_label_*"))
    for m in matches:
        suffix = m.name.split("_label_", 1)[-1]
        try:
            y = int(suffix)
        except ValueError:
            continue
        if y == y_true:
            return m
    return None


def run_dir(sample_dir: Path, run_id: int = 0) -> Path:
    return sample_dir / f"run_{run_id}"


def final_generation_csv(run_path: Path, run_id: int = 0) -> Path:
    return run_path / f"run_{run_id}_final_generation.csv"


def final_population_pickle(run_path: Path, run_id: int = 0) -> Path:
    return run_path / f"run_{run_id}_final_pop.pkl"
