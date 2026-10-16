"""Pool three runs into one classification JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

_GRACOIMAGE_ROOT = Path(__file__).resolve().parents[1]

GRACO_METHODS = ("graco_base_o2", "graco_base_o3", "graco50")
METHOD_LABELS = {
    "graco_base_o2": "CS_sparse",
    "graco_base_o3": "CS_prox",
    "graco50": "CS_50",
}
KEEP_NPZ_KEYS = ("orig_nhwc", "cf_graco_base_o2", "cf_graco_base_o3", "cf_graco50")
CURVE_KEYS = ("avg_o1", "avg_o2", "avg_o3", "min_o1", "min_o2", "min_o3")
CURVE_RUNS = (1, 2)
PER_SAMPLE_COLUMNS = (
    "has_valid_cf",
    "l2_distance",
    "sparsity_frac",
    "ssim_to_original",
    "lpips_to_original",
    "p_pred_cf",
    "ssim_to_random_pred_class",
    "lpips_to_random_pred_class",
    "inception_sqdist_to_source_class_mean",
    "inception_sqdist_to_predicted_class_mean",
)
MEAN_COLUMNS = tuple(c for c in PER_SAMPLE_COLUMNS if c != "has_valid_cf")

DATASETS = {
    "mnist": {
        "export": _GRACOIMAGE_ROOT / "MNIST" / "graco" / "export_graco_strategies.py",
        "eval": _GRACOIMAGE_ROOT / "common" / "eval_per_sample_mnist.py",
        "cmmd": _GRACOIMAGE_ROOT / "common" / "run_cmmd_mnist.py",
    },
    "cifar10": {
        "export": _GRACOIMAGE_ROOT / "Cifar10" / "graco" / "export_graco_strategies.py",
        "eval": _GRACOIMAGE_ROOT / "common" / "eval_per_sample_cifar10.py",
        "cmmd": _GRACOIMAGE_ROOT / "common" / "run_cmmd_cifar10.py",
    },
    "imagenet100": {
        "export": _GRACOIMAGE_ROOT / "Imagenet" / "graco" / "export_graco_strategies.py",
        "eval": _GRACOIMAGE_ROOT / "common" / "eval_per_sample_imagenet.py",
        "cmmd": _GRACOIMAGE_ROOT / "common" / "run_cmmd_imagenet.py",
    },
}


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mean_std(values: list[float]) -> tuple[float | None, float | None, int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None, 0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std, int(arr.size)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_GRACOIMAGE_ROOT))


def strip_baseline_keys(gen_root: Path) -> None:
    """Score only the three pooled CounterStroke picks."""
    for npz_path in sorted(gen_root.glob("sample_*/tensors.npz")):
        with np.load(npz_path) as blob:
            payload = {key: blob[key] for key in KEEP_NPZ_KEYS if key in blob.files}
        if "orig_nhwc" not in payload:
            raise RuntimeError(f"{npz_path} has no orig_nhwc")
        np.savez_compressed(npz_path, **payload)


def reduce_curves(graco_root: Path, run_ids: tuple[int, ...] = CURVE_RUNS) -> dict[str, Any]:
    """Mean and std across images of each objective, for the new runs only."""
    curves: dict[str, Any] = {}
    for run_id in run_ids:
        paths = sorted(graco_root.glob(f"**/run_{run_id}/objectives_by_generation.csv"))
        by_gen: dict[int, list[dict[str, str]]] = {}
        for path in paths:
            for row in _read_csv(path):
                by_gen.setdefault(int(row["gen"]), []).append(row)
        gens = sorted(by_gen)
        mean = {key: [] for key in CURVE_KEYS}
        std = {key: [] for key in CURVE_KEYS}
        n_at_gen = []
        for gen in gens:
            rows = by_gen[gen]
            n_at_gen.append(len(rows))
            for key in CURVE_KEYS:
                values = [v for v in (_finite(row.get(key)) for row in rows) if v is not None]
                m, s, _ = _mean_std(values)
                mean[key].append(m)
                std[key].append(s)
        curves[str(run_id)] = {
            "n_images": len(paths),
            "gen": gens,
            "n_at_gen": n_at_gen,
            "mean": mean,
            "std": std,
        }
        if not paths:
            print(f"warning: no objectives_by_generation.csv under {graco_root}/run_{run_id}", flush=True)
    return curves


def per_sample_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    picked = []
    for row in _read_csv(csv_path):
        method = row.get("method", "")
        if method not in GRACO_METHODS:
            continue
        item: dict[str, Any] = {
            "sample_idx": int(float(row["sample_idx"])),
            "y_true": _finite(row.get("y_true")),
            "method": method,
            "readout": METHOD_LABELS[method],
        }
        for column in PER_SAMPLE_COLUMNS:
            value = _finite(row.get(column))
            if column == "has_valid_cf" and value is not None:
                value = int(value)
            item[column] = value
        picked.append(item)
    return picked


def table_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One mean±std row per readout. Metrics use valid counterfactuals only."""
    out = []
    for method in GRACO_METHODS:
        group = [row for row in rows if row["method"] == method]
        valid = [row for row in group if row.get("has_valid_cf") == 1]
        flags = [float(row["has_valid_cf"]) for row in group if row.get("has_valid_cf") is not None]
        validity, _, n_flag = _mean_std(flags)
        entry: dict[str, Any] = {
            "method": method,
            "readout": METHOD_LABELS[method],
            "n": len(group),
            "n_valid": len(valid),
            "validity": validity if n_flag else None,
        }
        for column in MEAN_COLUMNS:
            mean, std, n = _mean_std([row[column] for row in valid if row.get(column) is not None])
            entry[f"{column}_mean"] = mean
            entry[f"{column}_std"] = std
            entry[f"{column}_n"] = n
        out.append(entry)
    return out


def distribution_rows(metrics_dir: Path) -> list[dict[str, Any]]:
    paired_path = metrics_dir / "distribution_metrics.csv"
    cmmd_path = metrics_dir / "cmmd_batch_stratified.csv"
    fid_path = metrics_dir / "batch_fid.csv"
    paired = {row["method"]: row for row in _read_csv(paired_path)} if paired_path.is_file() else {}
    cmmd = {row["method"]: row for row in _read_csv(cmmd_path)} if cmmd_path.is_file() else {}
    fid = {row["method"]: row for row in _read_csv(fid_path)} if fid_path.is_file() else {}
    rows = []
    for method in GRACO_METHODS:
        src = {}
        src.update(paired.get(method, {}))
        src.update(cmmd.get(method, {}))
        src.update(fid.get(method, {}))
        if not src:
            continue
        item: dict[str, Any] = {"method": method, "readout": METHOD_LABELS[method]}
        for key, value in src.items():
            if key in {"method", "note"}:
                continue
            number = _finite(value)
            item[key] = int(number) if key == "n_paired" and number is not None else number
        rows.append(item)
    return rows


def _export_and_score(name: str, base_gen: Path, graco_root: Path, side: Path, imagenet_root: Path | None) -> Path:
    spec = DATASETS[name]
    gen_root = side / name / "generation"
    metrics_dir = side / name
    gen_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(spec["export"]),
        "--base-gen-root",
        str(base_gen),
        "--graco-results-root",
        str(graco_root),
        "--out-gen-root",
        str(gen_root),
    ]
    _run(cmd)
    strip_baseline_keys(gen_root)
    eval_cmd = [
        sys.executable,
        str(spec["eval"]),
        "--gen-root",
        str(gen_root),
        "--csv-out",
        str(metrics_dir / "per_sample_metrics.csv"),
        "--batch-fid-csv",
        str(metrics_dir / "batch_fid.csv"),
    ]
    cmmd_cmd = [
        sys.executable,
        str(spec["cmmd"]),
        "--gen-root",
        str(gen_root),
        "--out-dir",
        str(metrics_dir),
    ]
    if name == "imagenet100":
        if imagenet_root is None:
            raise SystemExit("ImageNet scoring needs --imagenet-root")
        eval_cmd.extend(["--imagenet-root", str(imagenet_root)])
        cmmd_cmd.extend(["--imagenet-root", str(imagenet_root)])
    _run(eval_cmd)
    _run(cmmd_cmd)
    return metrics_dir


def build_dataset_block(
    graco_root: Path,
    metrics_dir: Path | None,
) -> dict[str, Any]:
    block: dict[str, Any] = {"curves": reduce_curves(graco_root)}
    if metrics_dir is None:
        return block
    rows = per_sample_rows(metrics_dir / "per_sample_metrics.csv")
    block["per_sample"] = rows
    block["table_means"] = table_means(rows)
    block["distribution"] = distribution_rows(metrics_dir)
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="JSON file to copy back. This is the only download.")
    parser.add_argument(
        "--side-root",
        type=Path,
        default=None,
        help="Rendered picks and metric CSVs. Default: <out parent>/multirun_side. Leave this on the GPU.",
    )
    parser.add_argument("--skip-score", action="store_true", help="Read curves only. Metrics CSVs must already exist under the side folder.")
    parser.add_argument("--mnist-base-gen", type=Path, default=None)
    parser.add_argument("--mnist-graco-results", type=Path, default=None)
    parser.add_argument("--cifar-base-gen", type=Path, default=None)
    parser.add_argument("--cifar-graco-results", type=Path, default=None)
    parser.add_argument("--imagenet-base-gen", type=Path, default=None)
    parser.add_argument("--imagenet-graco-results", type=Path, default=None)
    parser.add_argument("--imagenet-root", type=Path, default=None)
    args = parser.parse_args()

    specs = [
        ("mnist", args.mnist_base_gen, args.mnist_graco_results),
        ("cifar10", args.cifar_base_gen, args.cifar_graco_results),
        ("imagenet100", args.imagenet_base_gen, args.imagenet_graco_results),
    ]
    chosen = [(name, base, graco) for name, base, graco in specs if graco is not None]
    if not chosen:
        raise SystemExit("Pass at least one --*-graco-results directory.")

    side = args.side_root or (args.out.resolve().parent / "multirun_side")
    payload: dict[str, Any] = {
        "note": (
            "CS_sparse, CS_prox, and CS_50 are pooled picks over run_0, run_1, and run_2. "
            "Validity can only stay the same or rise, and sparsity and proximity can only "
            "improve versus run_0, because the picker sees a larger pool. "
            "This is not a standard error across seeds. "
            "Curves omit run_0 because that log mixed O1, O2, and O3 into one number."
        ),
        "methods": {method: METHOD_LABELS[method] for method in GRACO_METHODS},
        "datasets": {},
    }
    for name, base, graco in chosen:
        graco_root = graco.resolve()
        if not graco_root.is_dir():
            raise SystemExit(f"Missing Graco results directory: {graco_root}")
        metrics_dir = side / name
        if args.skip_score:
            if not (metrics_dir / "per_sample_metrics.csv").is_file():
                metrics_dir = None
        else:
            if base is None:
                raise SystemExit(f"{name} scoring needs the matching --*-base-gen directory.")
            metrics_dir = _export_and_score(name, base.resolve(), graco_root, side, args.imagenet_root)
        payload["datasets"][name] = build_dataset_block(graco_root, metrics_dir)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")
    print("Copy this JSON back. Leave the run folders and the side folder on the GPU.")


if __name__ == "__main__":
    main()
