"""Validate Graco / data paths for the setup wizard."""

from __future__ import annotations

from pathlib import Path

from .blank_paths import iter_target_classes, target_csv_path
from .datasets import AppDatasetSpec
from .metrics_paths import GEN_CSV_FILES, MetricsDirs


def count_factual_samples(graco_root: Path) -> int:
    if not graco_root.is_dir():
        return 0
    return sum(1 for p in graco_root.iterdir() if p.is_dir() and "_label_" in p.name)


def count_blank_targets(graco_run_root: Path, run_id: int = 0) -> int:
    if not graco_run_root.is_dir():
        return 0
    n = 0
    for k in iter_target_classes(graco_run_root):
        if target_csv_path(graco_run_root, k, run_id).is_file():
            n += 1
    return n


def validate_paths(
    spec: AppDatasetSpec,
    *,
    factual_graco: Path,
    blank_graco: Path,
    manifest_path: Path,
    data_root: Path,
    weights: Path,
    run_id: int,
) -> dict[str, tuple[bool, str]]:
    out: dict[str, tuple[bool, str]] = {}
    n_f = count_factual_samples(factual_graco)
    out["factual_graco"] = (
        factual_graco.is_dir() and n_f > 0,
        f"{n_f} sample folders" if n_f else "No `*_label_*` directories",
    )
    n_b = count_blank_targets(blank_graco, run_id)
    out["blank_graco"] = (
        blank_graco.is_dir() and n_b > 0,
        f"{n_b} target CSVs" if n_b else "No target_XXX/run_* CSV (optional for Factual tab)",
    )
    out["manifest"] = (
        manifest_path.is_file(),
        "Found" if manifest_path.is_file() else "Missing file",
    )
    out["data_root"] = (data_root.is_dir(), "OK" if data_root.is_dir() else "Not a directory")
    out["weights"] = (weights.is_file(), "Found" if weights.is_file() else "Missing weights")
    return out


def validate_metrics_paths(dirs: MetricsDirs) -> dict[str, tuple[bool, str]]:
    """Validate generation CSV dir and results plot dir for the metrics explorer."""
    out: dict[str, tuple[bool, str]] = {}
    out["gen_root"] = (
        dirs.gen_root.is_dir(),
        str(dirs.gen_root) if dirs.gen_root.is_dir() else "Not a directory",
    )
    n_csv = sum(1 for name in GEN_CSV_FILES if (dirs.gen_root / name).is_file())
    out["gen_csvs"] = (
        n_csv > 0,
        f"{n_csv}/{len(GEN_CSV_FILES)} CSV files present",
    )
    out["results_dir"] = (
        dirs.results_dir.is_dir(),
        str(dirs.results_dir) if dirs.results_dir.is_dir() else "Not a directory",
    )
    n_jpg = sum(1 for p in dirs.results_dir.glob("*.jpg")) if dirs.results_dir.is_dir() else 0
    out["plots"] = (
        n_jpg > 0,
        f"{n_jpg} JPG plot(s)" if n_jpg else "No JPG plots (run plot_methods / metrics_and_plots)",
    )
    summary = dirs.results_dir / "summary_tables.csv"
    out["summary_tables"] = (
        summary.is_file(),
        "Found" if summary.is_file() else "Missing (optional)",
    )
    return out
