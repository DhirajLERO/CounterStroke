"""Resolve metrics CSV and plot directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .datasets import AppDatasetSpec

GEN_CSV_FILES = (
    "per_sample_metrics.csv",
    "distribution_metrics.csv",
    "batch_fid.csv",
    "cmmd_per_sample.csv",
    "cmmd_batch_stratified.csv",
)

RESULTS_FILES = (
    "summary_tables.csv",
)


@dataclass(frozen=True)
class MetricsDirs:
    gen_root: Path
    results_dir: Path


def resolve_metrics_dirs(
    spec: AppDatasetSpec,
    *,
    gen_override: str | Path | None = None,
    results_override: str | Path | None = None,
) -> MetricsDirs:
    gen = Path(gen_override).expanduser() if gen_override else spec.default_gen_root
    results = (
        Path(results_override).expanduser()
        if results_override
        else spec.default_results_dir
    )
    return MetricsDirs(gen_root=gen.resolve(), results_dir=results.resolve())


def discover_available_files(dirs: MetricsDirs) -> dict[str, bool | list[str]]:
    gen_csvs = {name: (dirs.gen_root / name).is_file() for name in GEN_CSV_FILES}
    results_csv = (dirs.results_dir / "summary_tables.csv").is_file()
    plot_stems: list[str] = []
    if dirs.results_dir.is_dir():
        for p in dirs.results_dir.iterdir():
            if p.suffix.lower() == ".jpg":
                plot_stems.append(p.stem)
    plot_stems.sort()
    return {
        "gen_root_exists": dirs.gen_root.is_dir(),
        "results_dir_exists": dirs.results_dir.is_dir(),
        "gen_csvs": gen_csvs,
        "summary_tables": results_csv,
        "plot_stems": plot_stems,
    }
