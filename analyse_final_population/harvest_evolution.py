"""Harvest 30-run evolution traces + final Pareto clouds into static JSON."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

EVOLUTION_ROOT = Path(__file__).resolve().parents[1] / "evolution_experiment"
SITE_TO_EXP = {"mnist": "mnist", "cifar10": "cifar", "imagenet100": "imagenet"}
OBJ_KEYS = ("o1", "o2", "o3")
OBJ_LABELS = (
    {"id": "o1", "label": "O1 (validity)", "short": "O1"},
    {"id": "o2", "label": "O2 (sparsity)", "short": "O2"},
    {"id": "o3", "label": "O3 (proximity)", "short": "O3"},
)
PENALTY_CUTOFF = 1e5
UNIQUE_DECIMALS = 8
MAX_PARETO_POINTS = 4500


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")


def _flist(arr: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(arr, dtype=np.float64).ravel()]


def _load_trace(path: Path) -> dict[str, Any] | None:
    try:
        with np.load(path, allow_pickle=True) as z:
            best = np.asarray(z["best"], dtype=np.float64)
            gen = np.asarray(z["gen"], dtype=np.int64)
            sample_idx = int(np.asarray(z["sample_idx"]).reshape(-1)[0]) if "sample_idx" in z.files else -1
            y_true = int(np.asarray(z["y_true"]).reshape(-1)[0]) if "y_true" in z.files else -1
            run_id = int(np.asarray(z["run_id"]).reshape(-1)[0]) if "run_id" in z.files else -1
    except Exception:
        return None
    if best.ndim != 2 or best.shape[1] != 3:
        return None
    parent = path.parent
    sample_name = parent.parent.name
    if sample_idx < 0 and "_label_" in sample_name:
        left, right = sample_name.split("_label_", 1)
        sample_idx, y_true = int(left), int(right)
    if run_id < 0 and parent.name.startswith("run_"):
        try:
            run_id = int(parent.name.split("_", 1)[1])
        except ValueError:
            run_id = 0
    return {
        "gen": gen,
        "best": best,
        "sample_idx": sample_idx,
        "y_true": y_true,
        "run_id": run_id,
    }


def _align_best(traces: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    min_g = min(int(t["best"].shape[0]) for t in traces)
    gens = traces[0]["gen"][:min_g]
    stacked = np.stack([t["best"][:min_g] for t in traces], axis=0)
    return gens.astype(np.int64), stacked.astype(np.float64)


def _non_dominated_mask(pts: np.ndarray) -> np.ndarray:
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
    n = int(pts.shape[0])
    ranks = np.zeros(n, dtype=np.int32)
    remaining = np.arange(n, dtype=np.int64)
    rank = 1
    while remaining.size:
        mask = _non_dominated_mask(pts[remaining])
        ranks[remaining[mask]] = rank
        remaining = remaining[~mask]
        rank += 1
        if rank > n + 1:
            break
    return ranks


def _load_csv_objectives(path: Path) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"o1", "o2", "o3"}.issubset(set(reader.fieldnames)):
            return np.zeros((0, 3), dtype=np.float64)
        for rec in reader:
            try:
                rows.append((float(rec["o1"]), float(rec["o2"]), float(rec["o3"])))
            except (TypeError, ValueError):
                continue
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _curves_payload(
    *,
    dataset: str,
    sample_idx: int,
    y_true: int,
    gen: np.ndarray,
    best: np.ndarray,
) -> dict[str, Any]:
    mean = np.nanmean(best, axis=0)
    std = np.nanstd(best, axis=0, ddof=0)
    n_runs = int(best.shape[0])
    sem = std / max(n_runs, 1) ** 0.5
    runs = {
        k: [[float(best[r, g, i]) for g in range(best.shape[1])] for r in range(n_runs)]
        for i, k in enumerate(OBJ_KEYS)
    }
    return {
        "dataset": dataset,
        "sample_idx": int(sample_idx),
        "y_true": int(y_true),
        "n_runs": n_runs,
        "n_generations": int(gen.shape[0]),
        "objectives": list(OBJ_LABELS),
        "gen": [int(x) for x in gen],
        "mean": {k: _flist(mean[:, i]) for i, k in enumerate(OBJ_KEYS)},
        "std": {k: _flist(std[:, i]) for i, k in enumerate(OBJ_KEYS)},
        "sem": {k: _flist(sem[:, i]) for i, k in enumerate(OBJ_KEYS)},
        "runs": runs,
        "fixture": False,
    }


def _pareto_payload(
    *,
    dataset: str,
    sample_idx: int,
    y_true: int,
    n_runs: int,
    pts: np.ndarray,
) -> dict[str, Any] | None:
    if pts.size == 0:
        return None
    finite = np.isfinite(pts).all(axis=1)
    ok = finite & (pts[:, 1] < PENALTY_CUTOFF) & (pts[:, 2] < PENALTY_CUTOFF)
    clean = pts[ok]
    n_raw = int(pts.shape[0])
    n_clean = int(clean.shape[0])
    if n_clean == 0:
        return None
    unique = np.unique(np.round(clean, decimals=UNIQUE_DECIMALS), axis=0)
    if unique.shape[0] > MAX_PARETO_POINTS:
        rng = np.random.default_rng(0)
        keep = rng.choice(unique.shape[0], size=MAX_PARETO_POINTS, replace=False)
        unique = unique[np.sort(keep)]
    ranks = successive_ranks(unique)
    return {
        "dataset": dataset,
        "sample_idx": int(sample_idx),
        "y_true": int(y_true),
        "n_runs": int(n_runs),
        "n_raw": n_raw,
        "n_clean": n_clean,
        "n_unique": int(unique.shape[0]),
        "n_fronts": int(ranks.max()) if ranks.size else 0,
        "objectives": list(OBJ_LABELS),
        "points": [[float(a), float(b), float(c)] for a, b, c in unique],
        "rank": [int(r) for r in ranks],
        "fixture": False,
    }


def _from_traces(results_root: Path, dataset: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    traces = []
    for p in sorted(results_root.rglob("evolution_trace.npz")):
        t = _load_trace(p)
        if t is not None:
            traces.append(t)
    if not traces:
        return None, None
    by_sample: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        by_sample[int(t["sample_idx"])].append(t)
    chosen_id = max(by_sample, key=lambda k: len(by_sample[k]))
    chosen = sorted(by_sample[chosen_id], key=lambda t: t["run_id"])
    y_true = int(chosen[0]["y_true"])
    gen, best = _align_best(chosen)
    curves = _curves_payload(dataset=dataset, sample_idx=chosen_id, y_true=y_true, gen=gen, best=best)
    csvs = sorted(results_root.glob(f"{chosen_id}_label_{y_true}/run_*/run_*_final_generation.csv"))
    if not csvs:
        csvs = sorted(results_root.rglob("*_final_generation.csv"))
    chunks = [_load_csv_objectives(p) for p in csvs]
    chunks = [c for c in chunks if c.size]
    pareto = None
    if chunks:
        pts = np.concatenate(chunks, axis=0)
        pareto = _pareto_payload(
            dataset=dataset,
            sample_idx=chosen_id,
            y_true=y_true,
            n_runs=len(chosen),
            pts=pts,
        )
    return curves, pareto


def _from_plot_archives(data_dir: Path, dataset: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    curves = None
    pareto = None
    fit = data_dir / "single_example_best_fitness.npz"
    if fit.is_file():
        with np.load(fit, allow_pickle=True) as z:
            best = np.asarray(z["best"], dtype=np.float64)
            gen = np.asarray(z["gen"], dtype=np.int64)
            sample_idx = int(np.asarray(z["sample_idx"]).reshape(-1)[0]) if "sample_idx" in z.files else 0
            y_true = int(np.asarray(z["y_true"]).reshape(-1)[0]) if "y_true" in z.files else 0
        curves = _curves_payload(dataset=dataset, sample_idx=sample_idx, y_true=y_true, gen=gen, best=best)
    par = data_dir / "final_pareto_30runs.npz"
    if par.is_file():
        with np.load(par, allow_pickle=True) as z:
            pts = np.asarray(z["points"], dtype=np.float64)
            ranks = np.asarray(z["rank"], dtype=np.int32) if "rank" in z.files else successive_ranks(pts)
            sample_idx = int(np.asarray(z["sample_idx"]).reshape(-1)[0]) if "sample_idx" in z.files else 0
            y_true = int(np.asarray(z["y_true"]).reshape(-1)[0]) if "y_true" in z.files else 0
            n_runs = int(np.asarray(z["n_runs"]).reshape(-1)[0]) if "n_runs" in z.files else 30
        if pts.shape[0] > MAX_PARETO_POINTS:
            rng = np.random.default_rng(0)
            keep = np.sort(rng.choice(pts.shape[0], size=MAX_PARETO_POINTS, replace=False))
            pts = pts[keep]
            ranks = ranks[keep]
        pareto = {
            "dataset": dataset,
            "sample_idx": sample_idx,
            "y_true": y_true,
            "n_runs": n_runs,
            "n_raw": int(pts.shape[0]),
            "n_clean": int(pts.shape[0]),
            "n_unique": int(pts.shape[0]),
            "n_fronts": int(ranks.max()) if ranks.size else 0,
            "objectives": list(OBJ_LABELS),
            "points": [[float(a), float(b), float(c)] for a, b, c in pts],
            "rank": [int(r) for r in ranks],
            "fixture": False,
        }
    return curves, pareto


def write_evolution_fixture(out_root: Path) -> dict[str, Any]:
    index: dict[str, Any] = {}
    for ds, n_gen, n_runs in (("mnist", 100, 30), ("cifar10", 100, 30), ("imagenet100", 50, 30)):
        rng = np.random.default_rng(abs(hash(ds)) % 10_000)
        gen = np.arange(n_gen, dtype=np.int64)
        t = gen / max(n_gen - 1, 1)
        best = np.zeros((n_runs, n_gen, 3), dtype=np.float64)
        for r in range(n_runs):
            jitter = rng.normal(0, 0.04, n_gen)
            best[r, :, 0] = np.clip(-0.15 - 0.75 * (1 - np.exp(-3.2 * t)) + jitter * 0.15, -1.0, 0.05)
            best[r, :, 1] = np.clip(80 * np.exp(-2.4 * t) + 8 + rng.normal(0, 3.0, n_gen), 1.0, None)
            best[r, :, 2] = np.clip(12 * np.exp(-1.8 * t) + 1.2 + rng.normal(0, 0.4, n_gen), 0.2, None)
        curves = _curves_payload(
            dataset=ds,
            sample_idx=0,
            y_true=7 if ds != "imagenet100" else 0,
            gen=gen,
            best=best,
        )
        curves["fixture"] = True
        dest = out_root / "evolution" / ds
        _write_json(dest / "curves.json", curves)

        n_pts = 900
        u = rng.random(n_pts)
        pts = np.column_stack(
            [
                rng.uniform(-1.0, -0.05, n_pts),
                8 + 50 * u + rng.normal(0, 2, n_pts),
                1.2 + 7 * (1 - u) + rng.normal(0, 0.3, n_pts),
            ]
        )
        pareto = _pareto_payload(
            dataset=ds,
            sample_idx=curves["sample_idx"],
            y_true=curves["y_true"],
            n_runs=n_runs,
            pts=pts,
        )
        if pareto:
            pareto["fixture"] = True
            _write_json(dest / "pareto.json", pareto)
        index[ds] = {
            "curves": True,
            "pareto": bool(pareto),
            "sample_idx": curves["sample_idx"],
            "y_true": curves["y_true"],
            "n_runs": n_runs,
            "n_generations": n_gen,
            "fixture": True,
        }
    _write_json(out_root / "evolution" / "index.json", {"datasets": index})
    print("  [evolution] wrote fixture curves + Pareto clouds")
    return index


def harvest_evolution(
    out_root: Path,
    site_datasets: list[str],
    *,
    evolution_root: Path | None = None,
    write_fixture_if_missing: bool = False,
) -> dict[str, Any]:
    root = Path(evolution_root) if evolution_root is not None else EVOLUTION_ROOT
    index: dict[str, Any] = {}
    for site_ds in site_datasets:
        exp = SITE_TO_EXP.get(site_ds)
        if not exp:
            continue
        results_root = root / "results" / exp
        data_dir = root / "data" / exp
        curves, pareto = (None, None)
        if results_root.is_dir():
            curves, pareto = _from_traces(results_root, site_ds)
        if curves is None:
            c2, p2 = _from_plot_archives(data_dir, site_ds)
            curves = curves or c2
            pareto = pareto or p2
        dest = out_root / "evolution" / site_ds
        if curves:
            _write_json(dest / "curves.json", curves)
            print(f"  [evolution] {site_ds}: curves n_runs={curves['n_runs']} G={curves['n_generations']}")
        else:
            print(f"  [evolution] {site_ds}: no traces under {results_root}")
        if pareto:
            _write_json(dest / "pareto.json", pareto)
            print(f"  [evolution] {site_ds}: Pareto n_unique={pareto['n_unique']} fronts={pareto['n_fronts']}")
        src = curves or pareto
        if src:
            index[site_ds] = {
                "curves": bool(curves),
                "pareto": bool(pareto),
                "sample_idx": src.get("sample_idx"),
                "y_true": src.get("y_true"),
                "n_runs": src.get("n_runs"),
                "n_generations": (curves or {}).get("n_generations"),
                "fixture": bool(src.get("fixture")),
            }
    if not index and write_fixture_if_missing:
        return write_evolution_fixture(out_root)
    if index:
        _write_json(out_root / "evolution" / "index.json", {"datasets": index})
    return index


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Harvest 30-run evolution traces into static JSON")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--datasets", default="mnist,cifar10,imagenet100")
    p.add_argument("--evolution-root", type=Path, default=None)
    p.add_argument("--write-fixture", action="store_true")
    args = p.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    harvest_evolution(
        args.out.expanduser().resolve(),
        datasets,
        evolution_root=args.evolution_root,
        write_fixture_if_missing=args.write_fixture,
    )
