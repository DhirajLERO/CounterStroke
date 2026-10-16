"""Plot mean-best fitness or pooled successive Pareto fronts from 30-run CSVs."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_mpl_dir = _HERE / ".mplconfig"
_mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wrap_graco import (  # noqa: E402
    OBJECTIVE_LABELS,
    OBJECTIVE_NAMES,
    SPECS,
    data_dir_for,
    discover_final_generation_csvs,
    discover_traces,
    parse_sample_folder,
    plots_dir_for,
    results_dir_for,
)

N_OBJ = 3
PENALTY_CUTOFF = 1e5
UNIQUE_DECIMALS = 8
DEFAULT_MAX_VIVID_FRONTS = 4
_VIVID_ALPHA = {1: 0.90, 2: 0.55, 3: 0.40, 4: 0.25}
_VIVID_SIZE = {1: 36.0, 2: 22.0, 3: 16.0, 4: 12.0}


def _load_trace(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        best = np.asarray(z["best"], dtype=np.float64)
        gen = np.asarray(z["gen"], dtype=np.int64)
        sample_idx = int(np.asarray(z["sample_idx"]).reshape(-1)[0]) if "sample_idx" in z.files else -1
        y_true = int(np.asarray(z["y_true"]).reshape(-1)[0]) if "y_true" in z.files else -1
        seed = int(np.asarray(z["seed"]).reshape(-1)[0]) if "seed" in z.files else -1
        run_id = int(np.asarray(z["run_id"]).reshape(-1)[0]) if "run_id" in z.files else -1
    if best.ndim != 2 or best.shape[1] != N_OBJ:
        raise ValueError(f"Expected best with shape (G, 3) in {path}, got {best.shape}")
    parsed = parse_sample_folder(path.parent)
    if sample_idx < 0:
        sample_idx = parsed[0]
    if y_true < 0:
        y_true = parsed[1]
    if run_id < 0:
        run_id = parsed[2]
    return {
        "path": path,
        "gen": gen,
        "best": best,
        "sample_idx": sample_idx,
        "y_true": y_true,
        "seed": seed,
        "run_id": run_id,
    }


def _align_best(traces: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Return gen (G,) and best stacked as (N, G, 3), truncated to shared generations."""
    if not traces:
        raise ValueError("No evolution_trace.npz files to plot.")
    min_g = min(int(t["best"].shape[0]) for t in traces)
    if min_g < 1:
        raise ValueError("Traces have empty best arrays.")
    gens = traces[0]["gen"][:min_g]
    stacked = np.stack([t["best"][:min_g] for t in traces], axis=0)
    return gens.astype(np.int64), stacked.astype(np.float64)


def _mean_std(best: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(best, axis=0)
    std = np.nanstd(best, axis=0, ddof=0)
    return mean, std


def _plot_objective(
    gen: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    obj_idx: int,
    title: str,
    out_stem: Path,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = mean[:, obj_idx]
    s = std[:, obj_idx]
    ax.plot(gen, y, color="C0", lw=2.0, label="mean best")
    ax.fill_between(gen, y - s, y + s, color="C0", alpha=0.22, linewidth=0, label="±1 std")
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.ticklabel_format(useOffset=False, style="plain", axis="y")
    if obj_idx == 0:
        ax.set_ylim(-1.05, 0.05)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=160)
    plt.close(fig)


def _save_plot_archive(
    out_npz: Path,
    *,
    best: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    gen: np.ndarray,
    extra: dict[str, Any],
) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "best": np.asarray(best, dtype=np.float64),
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "gen": np.asarray(gen, dtype=np.int64),
        "objective_names": np.array(OBJECTIVE_NAMES, dtype=object),
    }
    payload.update(extra)
    np.savez_compressed(out_npz, **payload)


def plot_runs30(
    *,
    dataset: str,
    results_root: Path,
    plots_root: Path | None = None,
    data_root: Path | None = None,
    sample_idx: int | None = None,
) -> Path:
    traces = [_load_trace(p) for p in discover_traces(results_root)]
    if not traces:
        raise SystemExit(
            f"No evolution_trace.npz under {results_root}. "
            "Run evolution_experiment/run_experiment.py first."
        )

    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        by_sample[int(t["sample_idx"])].append(t)

    if sample_idx is not None and int(sample_idx) in by_sample:
        chosen_id = int(sample_idx)
    else:
        chosen_id = max(by_sample, key=lambda k: len(by_sample[k]))
    chosen = sorted(by_sample[chosen_id], key=lambda t: (t["run_id"], t["seed"]))
    y_true = int(chosen[0]["y_true"])

    gen, best = _align_best(chosen)
    mean, std = _mean_std(best)
    seeds = np.array([t["seed"] if t["seed"] >= 0 else t["run_id"] for t in chosen], dtype=np.int64)

    plots_root = plots_root or plots_dir_for(dataset)
    data_root = data_root or data_dir_for(dataset)
    title_base = f"{dataset}  sample {chosen_id} (label {y_true})  n={best.shape[0]} runs"
    for k, label in enumerate(OBJECTIVE_LABELS):
        _plot_objective(
            gen,
            mean,
            std,
            obj_idx=k,
            title=f"{title_base}\nmean best {label}",
            out_stem=plots_root / f"single_example_o{k + 1}",
            ylabel=f"Best {label}",
        )

    archive = data_root / "single_example_best_fitness.npz"
    _save_plot_archive(
        archive,
        best=best,
        mean=mean,
        std=std,
        gen=gen,
        extra={
            "seeds": seeds,
            "sample_idx": np.int64(chosen_id),
            "y_true": np.int64(y_true),
            "run_ids": np.array([t["run_id"] for t in chosen], dtype=np.int64),
        },
    )
    print(f"Wrote 30-run plots under {plots_root}")
    print(f"Wrote plot data {archive}  best.shape={best.shape}")
    return archive


def plot_examples100(
    *,
    dataset: str,
    results_root: Path,
    plots_root: Path | None = None,
    data_root: Path | None = None,
) -> Path:
    traces = [_load_trace(p) for p in discover_traces(results_root)]
    if not traces:
        raise SystemExit(
            f"No evolution_trace.npz under {results_root}. "
            "Stock turtle CSVs cannot be split into O1/O2/O3. "
            "Produce traces with evolution_experiment/run_experiment.py "
            "(e.g. --max-indices 100 --n-runs 1) and pass that results tree."
        )

    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        by_sample[int(t["sample_idx"])].append(t)

    per_example: list[np.ndarray] = []
    sample_ids: list[int] = []
    y_trues: list[int] = []
    n_runs_used: list[int] = []
    aligned_gen: np.ndarray | None = None
    min_g: int | None = None
    for sid in sorted(by_sample):
        group = by_sample[sid]
        gen_i, best_i = _align_best(group)
        g = int(best_i.shape[1])
        min_g = g if min_g is None else min(min_g, g)
        if aligned_gen is None or gen_i.shape[0] < aligned_gen.shape[0]:
            aligned_gen = gen_i
        # mean over runs for this example, shape (G, 3)
        per_example.append(np.nanmean(best_i, axis=0))
        sample_ids.append(sid)
        y_trues.append(int(group[0]["y_true"]))
        n_runs_used.append(int(best_i.shape[0]))

    assert aligned_gen is not None and min_g is not None
    truncated = [arr[:min_g] for arr in per_example]
    best = np.stack(truncated, axis=0)
    gen = aligned_gen[:min_g]
    mean, std = _mean_std(best)

    plots_root = plots_root or plots_dir_for(dataset)
    data_root = data_root or data_dir_for(dataset)
    title_base = f"{dataset}  mean over {best.shape[0]} examples"
    for k, label in enumerate(OBJECTIVE_LABELS):
        _plot_objective(
            gen,
            mean,
            std,
            obj_idx=k,
            title=f"{title_base}\nmean best {label}",
            out_stem=plots_root / f"examples100_o{k + 1}",
            ylabel=f"Best {label}",
        )

    archive = data_root / "examples100_best_fitness.npz"
    _save_plot_archive(
        archive,
        best=best,
        mean=mean,
        std=std,
        gen=gen,
        extra={
            "sample_ids": np.array(sample_ids, dtype=np.int64),
            "y_true": np.array(y_trues, dtype=np.int64),
            "n_runs_per_example": np.array(n_runs_used, dtype=np.int64),
        },
    )
    print(f"Wrote 100-example plots under {plots_root}")
    print(f"Wrote plot data {archive}  best.shape={best.shape}")
    return archive


def _load_csv_objectives(path: Path) -> np.ndarray:
    import csv

    rows: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"o1", "o2", "o3"}.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV missing o1/o2/o3 columns: {path}")
        for rec in reader:
            rows.append((float(rec["o1"]), float(rec["o2"]), float(rec["o3"])))
    if not rows:
        return np.zeros((0, N_OBJ), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _filter_penalty_rows(pts: np.ndarray) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, N_OBJ)
    finite = np.isfinite(pts).all(axis=1)
    ok = finite & (pts[:, 1] < PENALTY_CUTOFF) & (pts[:, 2] < PENALTY_CUTOFF)
    return pts[ok]


def _unique_points(pts: np.ndarray, *, decimals: int = UNIQUE_DECIMALS) -> np.ndarray:
    if pts.size == 0:
        return pts.reshape(0, N_OBJ)
    return np.unique(np.round(pts, decimals=decimals), axis=0)


def _non_dominated_mask(pts: np.ndarray) -> np.ndarray:
    """True where a point is not dominated (all three objectives minimized)."""
    n = int(pts.shape[0])
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        if not nd[i]:
            continue
        pi = pts[i]
        dominated = (
            (pi[0] <= pts[:, 0])
            & (pi[1] <= pts[:, 1])
            & (pi[2] <= pts[:, 2])
            & ((pi[0] < pts[:, 0]) | (pi[1] < pts[:, 1]) | (pi[2] < pts[:, 2]))
        )
        nd[dominated] = False
        nd[i] = True
    return nd


def successive_ranks(pts: np.ndarray) -> np.ndarray:
    """NSGA-II-style ranks: 1 = first front, then peel the rest."""
    n = int(pts.shape[0])
    ranks = np.zeros(n, dtype=np.int32)
    if n == 0:
        return ranks
    remaining = np.arange(n, dtype=np.int64)
    rank = 1
    while remaining.size:
        mask = _non_dominated_mask(pts[remaining])
        ranks[remaining[mask]] = rank
        remaining = remaining[~mask]
        rank += 1
        if rank > n + 1:
            raise RuntimeError("Successive ranking failed to exhaust points.")
    return ranks


def _choose_sample_csvs(
    records: list[dict[str, Any]],
    sample_idx: int | None,
) -> tuple[int, int, list[dict[str, Any]]]:
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_sample[int(rec["sample_idx"])].append(rec)
    if sample_idx is not None and int(sample_idx) in by_sample:
        chosen_id = int(sample_idx)
    else:
        chosen_id = max(by_sample, key=lambda k: len(by_sample[k]))
    chosen = sorted(by_sample[chosen_id], key=lambda r: int(r["run_id"]))
    y_true = int(chosen[0]["y_true"])
    return chosen_id, y_true, chosen


def _front_style(rank: int, max_vivid: int) -> dict[str, Any]:
    if rank > max_vivid:
        return {
            "c": "#2f6fad",
            "s": 8.0,
            "alpha": 0.22,
            "label": "Rest",
            "zorder": 1,
        }
    t = (max_vivid - rank) / max(1, max_vivid - 1)
    cmap = plt.cm.plasma
    alpha = _VIVID_ALPHA.get(rank)
    size = _VIVID_SIZE.get(rank)
    if alpha is None:
        alpha = 0.90 - (rank - 1) / max(1, max_vivid - 1) * (0.90 - 0.25)
    if size is None:
        size = 36.0 - (rank - 1) / max(1, max_vivid - 1) * (36.0 - 12.0)
    return {
        "c": cmap(0.25 + 0.65 * t),
        "s": float(size),
        "alpha": float(alpha),
        "label": f"Front {rank}",
        "zorder": 10 + (max_vivid - rank),
    }


def _apply_o1_limit(ax: Any, axis: str) -> None:
    if axis == "x":
        ax.set_xlim(-1.05, 0.05)
    elif axis == "y":
        ax.set_ylim(-1.05, 0.05)
    elif axis == "z":
        ax.set_zlim(-1.05, 0.05)
    ax.ticklabel_format(useOffset=False, style="plain", axis=axis)


def _padded_limits(values: np.ndarray, *, lo: float | None = None, pad: float = 0.06) -> tuple[float, float]:
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if lo is not None:
        vmin = min(vmin, lo)
    span = max(vmax - vmin, 1e-6)
    return vmin - pad * span, vmax + pad * span


def _paper_rc() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _legend_inside(
    ax: Any,
    *,
    loc: str = "lower left",
    bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    priority = {f"Front {i}": i for i in range(1, 16)}
    priority["Projection"] = 90
    priority["Rest"] = 100
    ordered = sorted(zip(handles, labels), key=lambda hl: priority.get(hl[1], 50))
    handles, labels = zip(*ordered)
    kwargs: dict[str, Any] = dict(
        loc=loc,
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        framealpha=0.92,
        borderpad=0.35,
        handletextpad=0.4,
        markerscale=1.1,
    )
    if bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
        kwargs["borderaxespad"] = 0.0
    ax.legend(handles, labels, **kwargs)


def _style_3d_box(ax: Any) -> None:
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.55, 0.55, 0.55, 1.0))
    ax.grid(True, color="0.75", alpha=0.55, linewidth=0.6)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def _project_onto_planes(
    ax: Any,
    points: np.ndarray,
    ranks: np.ndarray,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    max_vivid: int,
) -> None:
    """Gray shadows of the vivid fronts only — colour stays reserved for the 3D layers."""
    sel = (ranks >= 1) & (ranks <= max_vivid)
    pts = points[sel]
    n = int(pts.shape[0])
    if n == 0:
        return
    x_wall, y_wall, z_floor = xlim[0], ylim[1], zlim[0]
    kw = dict(
        s=9.0,
        marker="s",
        color="0.35",
        alpha=0.22,
        linewidths=0,
        depthshade=False,
        zorder=0,
    )
    ax.scatter(pts[:, 0], pts[:, 1], np.full(n, z_floor), label="Projection", **kw)
    ax.scatter(pts[:, 0], np.full(n, y_wall), pts[:, 2], **kw)
    ax.scatter(np.full(n, x_wall), pts[:, 1], pts[:, 2], **kw)


def _scatter_fronts(
    ax: Any,
    points: np.ndarray,
    ranks: np.ndarray,
    *,
    x: int,
    y: int,
    z: int | None,
    max_vivid: int,
) -> None:
    max_rank = int(ranks.max()) if ranks.size else 0
    rest = ranks > max_vivid
    if rest.any():
        style = _front_style(max_vivid + 1, max_vivid)
        kwargs = dict(
            s=style["s"],
            color=style["c"],
            alpha=style["alpha"],
            linewidths=0,
            label=style["label"],
            zorder=style["zorder"],
        )
        if z is None:
            ax.scatter(points[rest, x], points[rest, y], **kwargs)
        else:
            ax.scatter(
                points[rest, x],
                points[rest, y],
                points[rest, z],
                depthshade=False,
                **kwargs,
            )
    for rank in range(min(max_rank, max_vivid), 0, -1):
        sel = ranks == rank
        if not sel.any():
            continue
        style = _front_style(rank, max_vivid)
        kwargs = dict(
            s=style["s"],
            color=style["c"],
            alpha=style["alpha"],
            linewidths=0.15,
            edgecolors="none",
            label=style["label"],
            zorder=style["zorder"],
        )
        if z is None:
            ax.scatter(points[sel, x], points[sel, y], **kwargs)
        else:
            ax.scatter(
                points[sel, x],
                points[sel, y],
                points[sel, z],
                depthshade=False,
                **kwargs,
            )


def _save_figure(fig: Any, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.25)
    fig.savefig(out_stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _plot_pareto_3d(
    points: np.ndarray,
    ranks: np.ndarray,
    *,
    out_stem: Path,
    max_vivid: int,
) -> None:
    xlim = (-1.05, 0.05)
    ylim = _padded_limits(points[:, 1])
    zlim = _padded_limits(points[:, 2])
    with plt.rc_context(_paper_rc()):
        fig = plt.figure(figsize=(7.2, 6.2))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        _style_3d_box(ax)
        _project_onto_planes(
            ax, points, ranks, xlim=xlim, ylim=ylim, zlim=zlim, max_vivid=max_vivid
        )
        _scatter_fronts(points=points, ranks=ranks, ax=ax, x=0, y=1, z=2, max_vivid=max_vivid)
        ax.set_xlabel(OBJECTIVE_LABELS[0], labelpad=14)
        ax.set_ylabel(OBJECTIVE_LABELS[1], labelpad=14)
        ax.set_zlabel("")
        ax.text(
            xlim[0] - 0.02,
            ylim[0] - 0.04 * (ylim[1] - ylim[0]),
            0.52 * (zlim[0] + zlim[1]),
            OBJECTIVE_LABELS[2],
            zdir="z",
            ha="center",
            va="center",
        )
        _apply_o1_limit(ax, "x")
        ax.view_init(elev=22, azim=-50)
        _legend_inside(ax, loc="center left", bbox_to_anchor=(1.04, 0.52))
        fig.subplots_adjust(left=0.06, right=0.78, bottom=0.08, top=0.96)
        _save_figure(fig, out_stem)


def _plot_pareto_pair(
    points: np.ndarray,
    ranks: np.ndarray,
    *,
    i: int,
    j: int,
    out_stem: Path,
    max_vivid: int,
) -> None:
    with plt.rc_context(_paper_rc()):
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        _scatter_fronts(points=points, ranks=ranks, ax=ax, x=i, y=j, z=None, max_vivid=max_vivid)
        ax.set_xlabel(OBJECTIVE_LABELS[i], labelpad=6)
        ax.set_ylabel(OBJECTIVE_LABELS[j], labelpad=6)
        if i == 0:
            _apply_o1_limit(ax, "x")
        if j == 0:
            _apply_o1_limit(ax, "y")
        ax.grid(True, color="0.8", alpha=0.55, linewidth=0.6)
        ax.set_axisbelow(True)
        _legend_inside(ax, loc="upper right")
        fig.tight_layout()
        _save_figure(fig, out_stem)


def plot_pareto30(
    *,
    dataset: str,
    results_root: Path,
    plots_root: Path | None = None,
    data_root: Path | None = None,
    sample_idx: int | None = None,
    max_vivid_fronts: int = DEFAULT_MAX_VIVID_FRONTS,
) -> Path:
    paths = discover_final_generation_csvs(results_root)
    if not paths:
        raise SystemExit(
            f"No *_final_generation.csv under {results_root}. "
            "Run evolution_experiment/run_experiment.py first."
        )

    records: list[dict[str, Any]] = []
    for path in paths:
        sample_id, y_true, run_id = parse_sample_folder(path.parent)
        records.append(
            {
                "path": path,
                "sample_idx": sample_id,
                "y_true": y_true,
                "run_id": run_id,
            }
        )
    chosen_id, y_true, chosen = _choose_sample_csvs(records, sample_idx)
    if sample_idx is not None and int(sample_idx) != chosen_id:
        raise SystemExit(
            f"sample-idx {sample_idx} not found under {results_root}; "
            f"available={[k for k in sorted({int(r['sample_idx']) for r in records})]}"
        )

    raw_chunks = [_load_csv_objectives(r["path"]) for r in chosen]
    raw = (
        np.concatenate(raw_chunks, axis=0)
        if raw_chunks
        else np.zeros((0, N_OBJ), dtype=np.float64)
    )
    cleaned = _filter_penalty_rows(raw)
    points = _unique_points(cleaned)
    if points.shape[0] == 0:
        raise SystemExit(f"No finite non-penalty (o1,o2,o3) points in {len(chosen)} CSVs.")
    ranks = successive_ranks(points)
    max_vivid = max(1, int(max_vivid_fronts))

    plots_root = plots_root or plots_dir_for(dataset)
    data_root = data_root or data_dir_for(dataset)
    n_fronts = int(ranks.max())
    _plot_pareto_3d(
        points,
        ranks,
        out_stem=plots_root / "pareto_3d",
        max_vivid=max_vivid,
    )
    pairs = ((0, 1, "pareto_o1_o2"), (0, 2, "pareto_o1_o3"), (1, 2, "pareto_o2_o3"))
    for i, j, stem in pairs:
        _plot_pareto_pair(
            points,
            ranks,
            i=i,
            j=j,
            out_stem=plots_root / stem,
            max_vivid=max_vivid,
        )

    archive = data_root / "final_pareto_30runs.npz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive,
        points=points.astype(np.float64),
        rank=ranks.astype(np.int32),
        n_unique=np.int64(points.shape[0]),
        n_raw=np.int64(raw.shape[0]),
        n_clean=np.int64(cleaned.shape[0]),
        sample_idx=np.int64(chosen_id),
        y_true=np.int64(y_true),
        n_runs=np.int64(len(chosen)),
        n_fronts=np.int64(n_fronts),
        max_vivid_fronts=np.int64(max_vivid),
        objective_names=np.array(OBJECTIVE_NAMES, dtype=object),
        run_ids=np.array([int(r["run_id"]) for r in chosen], dtype=np.int64),
    )
    print(f"Wrote Pareto plots under {plots_root}")
    print(
        f"Wrote {archive}  unique={points.shape[0]} raw={raw.shape[0]} "
        f"fronts={n_fronts} runs={len(chosen)}"
    )
    return archive


def _default_results_root(dataset: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    local = results_dir_for(dataset)
    if local.is_dir() and discover_traces(local):
        return local
    return local


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot mean-best fitness or pooled successive Pareto fronts."
    )
    p.add_argument("--mode", choices=("runs30", "examples100", "pareto30"), required=True)
    p.add_argument("--dataset", choices=tuple(SPECS), required=True)
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Root that contains {idx}_label_{y}/run_*/evolution_trace.npz "
        "and run_*_final_generation.csv "
        "(default: evolution_experiment/results/<dataset>).",
    )
    p.add_argument(
        "--sample-idx",
        type=int,
        default=None,
        help="runs30 / pareto30: which example if several exist",
    )
    p.add_argument(
        "--max-vivid-fronts",
        type=int,
        default=DEFAULT_MAX_VIVID_FRONTS,
        help="pareto30: how many successive fronts stay coloured (rest is a faint cloud).",
    )
    args = p.parse_args()

    results_root = _default_results_root(args.dataset, args.results_root)
    if args.mode == "runs30":
        plot_runs30(dataset=args.dataset, results_root=results_root, sample_idx=args.sample_idx)
    elif args.mode == "examples100":
        plot_examples100(dataset=args.dataset, results_root=results_root)
    else:
        plot_pareto30(
            dataset=args.dataset,
            results_root=results_root,
            sample_idx=args.sample_idx,
            max_vivid_fronts=args.max_vivid_fronts,
        )


if __name__ == "__main__":
    main()
