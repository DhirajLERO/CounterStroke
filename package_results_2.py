#!/usr/bin/env python3
"""Extract paper-ready result CSVs with full per-sample data for significance tests."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_results"

DATASETS = {
    "MNIST": ROOT / "MNIST" / "outputs" / "generation",
    "CIFAR-10": ROOT / "Cifar10" / "outputs" / "generation",
    "ImageNet-100": ROOT / "Imagenet" / "outputs" / "generation",
}

GRACO_METHODS = {"graco_base_o2", "graco_base_o3", "graco50"}

# Keep every numeric metric that may appear in significance tests / tables.
METRIC_CANDIDATES = [
    "l2_distance",
    "sparsity_count",
    "sparsity_frac",
    "ssim_to_original",
    "ssim_to_random_pred_class",
    "lpips_to_original",
    "lpips_to_random_pred_class",
    "p_true_orig",
    "p_true_cf",
    "delta_p_true",
    "p_intended_target_cf",
    "p_pred_cf",
    "inception_sqdist_to_source_class_mean",
    "inception_sqdist_to_predicted_class_mean",
]


def load_graco_meta(gen_root: Path) -> dict[int, dict]:
    """sample_idx -> meta fields needed for validity / success definitions."""
    out: dict[int, dict] = {}
    for meta_path in sorted(gen_root.glob("sample_*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = meta.get("sample_idx", meta.get("dataset_index"))
        if sid is None:
            try:
                sid = int(meta_path.parent.name.split("_", 1)[1])
            except Exception:
                continue
        out[int(sid)] = {
            "graco_strategy_valid": meta.get("graco_strategy_valid", {}),
            "graco_selected_row": meta.get("graco_selected_row", {}),
            "methods": meta.get("methods", {}),
        }
    return out


def graco_success_from_meta(meta_row: dict, method: str) -> bool | None:
    """Return True/False if o1 available; else None."""
    gsr = meta_row.get("graco_selected_row") or {}
    ent = gsr.get(method)
    if not isinstance(ent, dict):
        return None
    try:
        return float(ent.get("o1")) < 0.0
    except Exception:
        return None


def prepare_dataset(dataset: str, gen_root: Path) -> pd.DataFrame:
    path = gen_root / "per_sample_metrics.csv"
    if not path.is_file():
        print(f"WARNING: missing {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.insert(0, "dataset", dataset)

    # Force numeric where possible (keeps NaNs for missing CFs).
    for col in df.columns:
        if col in {"dataset", "method"}:
            continue
        if col.startswith("note"):
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")

    meta_map = load_graco_meta(gen_root)

    available = []
    class_flip = []
    target_hit = []
    success_plot = []  # paper success: flip for baselines; o1<0 for Graco when available

    for _, row in df.iterrows():
        method = str(row["method"])
        sid = int(row["sample_idx"])
        meta = meta_map.get(sid, {})

        # Availability
        has_valid = int(row.get("has_valid_cf", 0) or 0)
        if method in GRACO_METHODS:
            gsv = meta.get("graco_strategy_valid") or {}
            if method in gsv:
                has_valid = int(bool(gsv[method]))
        available.append(has_valid)

        pred = row.get("pred_class_cf")
        y_true = row.get("y_true")
        y_tgt = row.get("y_intended_target")
        pred_ok = pd.notna(pred)
        avail = bool(has_valid)

        flip = bool(avail and pred_ok and int(pred) != int(y_true))
        hit = bool(avail and pred_ok and int(pred) == int(y_tgt))
        class_flip.append(int(flip))
        target_hit.append(int(hit))

        if method in GRACO_METHODS:
            g_succ = graco_success_from_meta(meta, method)
            success_plot.append(int(g_succ) if g_succ is not None else int(flip))
        else:
            success_plot.append(int(flip))

    df["available"] = available
    df["class_flip"] = class_flip
    df["target_hit"] = target_hit
    df["success_plot"] = success_plot
    return df


def mean_summary(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(values))
    if n == 0:
        return {
            "n": 0, "mean": math.nan, "std": math.nan, "median": math.nan,
            "sem": math.nan, "ci95_low": math.nan, "ci95_high": math.nan,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(values.median()),
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
    }


def build_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), g in df.groupby(["dataset", "method"], sort=False):
        n = len(g)
        rows.append({
            "dataset": dataset,
            "method": method,
            "n_attempted": n,
            "n_available": int(g["available"].sum()),
            "availability_rate": float(g["available"].mean()),
            "n_class_flip": int(g["class_flip"].sum()),
            "class_flip_validity": float(g["class_flip"].mean()),
            "n_target_hit": int(g["target_hit"].sum()),
            "target_attainment": float(g["target_hit"].mean()),
            "n_success_plot": int(g["success_plot"].sum()),
            "success_plot_rate": float(g["success_plot"].mean()),
        })
    return pd.DataFrame(rows)


def build_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRIC_CANDIDATES if m in df.columns]
    rows = []
    subsets = {
        "all_attempted": pd.Series(True, index=df.index),
        "available": df["available"].eq(1),
        "successful_flip": df["class_flip"].eq(1),
        "target_hit": df["target_hit"].eq(1),
        "success_plot": df["success_plot"].eq(1),
    }
    for subset_name, mask in subsets.items():
        sub = df.loc[mask]
        for (dataset, method), g in sub.groupby(["dataset", "method"], sort=False):
            for metric in metrics:
                rows.append({
                    "dataset": dataset,
                    "method": method,
                    "subset": subset_name,
                    "metric": metric,
                    **mean_summary(g[metric]),
                })
    return pd.DataFrame(rows)


def collect_optional_csv(name: str) -> pd.DataFrame:
    frames = []
    for dataset, gen_root in DATASETS.items():
        path = gen_root / name
        if not path.is_file():
            print(f"WARNING: missing {path}")
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", dataset)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_explainability() -> None:
    """Copy SRG per-sample tables if present (also useful for significance tests)."""
    src_root = ROOT / "explainability" / "outputs"
    if not src_root.is_dir():
        return
    dst = OUT / "explainability"
    dst.mkdir(parents=True, exist_ok=True)
    for dataset_dir in ["mnist", "cifar10", "imagenet100"]:
        for fname in ["srg_summary.csv", "method_ranking.csv"]:
            src = src_root / dataset_dir / fname
            if src.is_file():
                shutil.copy2(src, dst / f"{dataset_dir}_{fname}")
    for fname in ["cross_dataset_srg.csv"]:
        src = src_root / fname
        if src.is_file():
            shutil.copy2(src, dst / fname)


def collect_benchmark() -> None:
    dst = OUT / "benchmark"
    dst.mkdir(parents=True, exist_ok=True)
    for rel in [
        "benchmark/morphomnist/outputs/summary.csv",
        "benchmark/celeba/outputs/summary.csv",
    ]:
        src = ROOT / rel
        if src.is_file():
            shutil.copy2(src, dst / src.name.replace("summary.csv", src.parts[-3] + "_summary.csv")
                         if False else Path(rel).name)
            # clearer names:
    morpho = ROOT / "benchmark/morphomnist/outputs/summary.csv"
    celeba = ROOT / "benchmark/celeba/outputs/summary.csv"
    if morpho.is_file():
        shutil.copy2(morpho, dst / "morphomnist_summary.csv")
    if celeba.is_file():
        shutil.copy2(celeba, dst / "celeba_summary.csv")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    for dataset, gen_root in DATASETS.items():
        frame = prepare_dataset(dataset, gen_root)
        if frame.empty:
            continue
        frames.append(frame)
        # Also keep dataset-local raw file untouched for debugging.
        frame.to_csv(OUT / f"per_sample_{dataset.replace('-', '_').replace(' ', '_')}.csv",
                     index=False)

    if not frames:
        raise SystemExit("No per_sample_metrics.csv files found.")

    combined = pd.concat(frames, ignore_index=True)

    # PRIMARY file for significance testing: every sample × method row.
    combined.to_csv(OUT / "main_per_sample_all_datasets.csv", index=False)

    build_rate_table(combined).to_csv(OUT / "main_validity_summary.csv", index=False)
    build_metric_summary(combined).to_csv(OUT / "main_metric_summary.csv", index=False)

    # Distribution / CMMD / FID tables (method-level, still useful).
    for src_name, dst_name in [
        ("distribution_metrics.csv", "main_distribution_metrics.csv"),
        ("batch_fid.csv", "main_batch_fid.csv"),
        ("cmmd_per_sample.csv", "main_cmmd_per_sample.csv"),
        ("cmmd_batch_stratified.csv", "main_cmmd_batch_stratified.csv"),
    ]:
        frame = collect_optional_csv(src_name)
        if not frame.empty:
            frame.to_csv(OUT / dst_name, index=False)

    collect_explainability()
    collect_benchmark()

    # Manifest of what was packaged.
    manifest = {
        "n_rows_per_sample": int(len(combined)),
        "datasets": sorted(combined["dataset"].unique().tolist()),
        "methods": sorted(combined["method"].unique().tolist()),
        "columns": combined.columns.tolist(),
        "files": sorted(p.name for p in OUT.rglob("*") if p.is_file()),
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = shutil.make_archive(str(ROOT / "paper_results_main"), "zip", root_dir=OUT)
    print(f"Wrote: {OUT}")
    print(f"Zip:   {zip_path}")
    print(f"Per-sample rows: {len(combined)}")
    print("Download only: paper_results_main.zip")


if __name__ == "__main__":
    main()