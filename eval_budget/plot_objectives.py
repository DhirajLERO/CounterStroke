"""Mean best-objective curves for the 300-generation budget runs."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from snapshots import mask_penalty
from spec import MPLCONFIG, OBJECTIVE_COLUMNS, SNAPSHOT_GENS, plots_root, runs_root

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
MPLCONFIG.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "min_o1": "O1 (validity)",
    "min_o2": "O2 (sparsity)",
    "min_o3": "O3 (proximity)",
}


def _read_curve(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    gen = np.array([int(float(row["gen"])) for row in rows], dtype=np.int64)
    columns = {}
    for name in OBJECTIVE_COLUMNS:
        columns[name] = np.array([float(row[name]) for row in rows], dtype=np.float64)
    return gen, columns


def plot_dataset(dataset: str) -> None:
    root = runs_root(dataset)
    grouped: dict[str, list[tuple[np.ndarray, dict[str, np.ndarray]]]] = defaultdict(list)
    if root.is_dir():
        for path in sorted(root.glob("*_label_*/run_*/objectives_by_generation.csv")):
            loaded = _read_curve(path)
            if loaded is None:
                continue
            grouped[path.parents[1].name].append(loaded)
    if not grouped:
        print(f"  no objective logs yet under {root}", flush=True)
        return

    per_sample: list[dict[str, np.ndarray]] = []
    gen_ref: np.ndarray | None = None
    min_len: int | None = None
    for traces in grouped.values():
        if len(traces) < 1:
            continue
        length = min(int(gen.shape[0]) for gen, _cols in traces)
        if length < 1:
            continue
        min_len = length if min_len is None else min(min_len, length)
        if gen_ref is None or traces[0][0].shape[0] < gen_ref.shape[0]:
            gen_ref = traces[0][0]
        stacked: dict[str, np.ndarray] = {}
        for name in ("min_o1", "min_o2", "min_o3"):
            runs = np.stack([cols[name][:length] for _gen, cols in traces], axis=0)
            if name != "min_o1":
                runs = mask_penalty(runs)
            stacked[name] = np.nanmean(runs, axis=0)
        per_sample.append(stacked)
    if not per_sample or gen_ref is None or min_len is None:
        print(f"  objective logs under {root} are empty", flush=True)
        return

    gen = gen_ref[:min_len]
    out = plots_root(dataset)
    out.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {"gen": gen.astype(np.int64)}
    for name, label in LABELS.items():
        stack = np.stack([sample[name][:min_len] for sample in per_sample], axis=0)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
        archive[f"{name}_mean"] = mean
        archive[f"{name}_std"] = std
        _draw(gen, mean, std, title=f"{dataset}  n={stack.shape[0]} samples", ylabel=f"Best {label}", stem=out / name)
    np.savez_compressed(out / "objective_curves.npz", **archive)
    print(f"Wrote objective curves under {out}", flush=True)


def _draw(gen: np.ndarray, mean: np.ndarray, std: np.ndarray, *, title: str, ylabel: str, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(gen, mean, color="C0", lw=2.0, label="mean across samples")
    ax.fill_between(gen, mean - std, mean + std, color="C0", alpha=0.22, linewidth=0, label="±1 std")
    for generation in SNAPSHOT_GENS:
        if gen.size and int(generation) <= int(gen[-1]):
            ax.axvline(generation, color="0.45", lw=0.8, ls="--")
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)
