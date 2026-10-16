"""Aggregate SRG plots and CSV summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..paths import OUTPUTS_ROOT


def collect_srg_rows(dataset: str) -> pd.DataFrame:
    root = OUTPUTS_ROOT / dataset
    rows: list[dict] = []
    if not root.is_dir():
        return pd.DataFrame(rows)
    for sample_dir in sorted(root.glob("sample_*")):
        meta_path = sample_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for method, score in (meta.get("srg_scores") or {}).items():
            rows.append(
                {
                    "dataset": dataset,
                    "sample_idx": meta.get("sample_idx"),
                    "y_true": meta.get("y_true"),
                    "explanation_class": meta.get("explanation_class"),
                    "n_phenotypes": meta.get("n_phenotypes"),
                    "method": method,
                    "srg_score": score,
                }
            )
    return pd.DataFrame(rows)


def srg_method_ranking(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["method", "mean", "std", "count"])
    return (
        df.groupby("method")["srg_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def format_srg_mean_table(ranking: pd.DataFrame, *, title: str | None = None) -> str:
    lines: list[str] = []
    if title:
        lines.append(title)
    if ranking.empty:
        lines.append("(no SRG scores)")
        return "\n".join(lines)

    headers = ("method", "mean", "std", "n")
    rows: list[tuple[str, str, str, str]] = []
    for _, row in ranking.iterrows():
        std = row["std"]
        rows.append(
            (
                str(row["method"]),
                f"{row['mean']:.4f}",
                f"{std:.4f}" if pd.notna(std) else "—",
                str(int(row["count"])),
            )
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[i] for i in range(len(headers)))
    lines.extend([header_line, sep_line])
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def print_srg_mean_table(
    dataset: str,
    *,
    file=None,
) -> pd.DataFrame:
    """Print mean SRG per method for a dataset; returns the ranking DataFrame."""
    df = collect_srg_rows(dataset)
    ranking = srg_method_ranking(df)
    n_samples = int(df["sample_idx"].nunique()) if not df.empty else 0
    table = format_srg_mean_table(
        ranking,
        title=f"Mean SRG by method — {dataset} (n={n_samples} samples)",
    )
    print(table, file=file or sys.stdout)
    return ranking


def print_cross_dataset_srg_table(datasets: list[str], *, file=None) -> None:
    """Print mean SRG per method, columns = datasets."""
    frames = [collect_srg_rows(d) for d in datasets]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = file or sys.stdout
    print("", file=out)
    if combined.empty:
        print("Mean SRG by method (cross-dataset)\n(no SRG scores)", file=out)
        return

    pivot = combined.groupby(["method", "dataset"])["srg_score"].mean().unstack("dataset")
    pivot = pivot.reindex(columns=[d for d in datasets if d in pivot.columns])
    first_col = pivot.columns[0]
    pivot = pivot.sort_values(first_col, ascending=False, na_position="last")
    formatted = pivot.map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    print("Mean SRG by method (cross-dataset)", file=out)
    print(formatted.to_string(), file=out)


def write_dataset_summary(dataset: str) -> Path:
    df = collect_srg_rows(dataset)
    out_csv = OUTPUTS_ROOT / dataset / "srg_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if df.empty:
        return out_csv

    ranking = srg_method_ranking(df)
    ranking.to_csv(OUTPUTS_ROOT / dataset / "method_ranking.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = ranking["method"].tolist()
    means = ranking["mean"].tolist()
    stds = ranking["std"].fillna(0).tolist()
    ax.bar(methods, means, yerr=stds, capsize=3)
    ax.set_ylabel("Mean SRG")
    ax.set_title(f"SRG by method — {dataset}")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUTS_ROOT / dataset / "srg_by_method_box.png", dpi=150)
    plt.close(fig)

    _plot_mean_curves(dataset, df)
    return out_csv


def _plot_mean_curves(dataset: str, df: pd.DataFrame) -> None:
    methods = sorted(df["method"].unique())
    n_methods = len(methods)
    if n_methods == 0:
        return
    cols = min(3, n_methods)
    rows = int(np.ceil(n_methods / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for i, method in enumerate(methods):
        ax = axes[i // cols][i % cols]
        mif_list: list[np.ndarray] = []
        lif_list: list[np.ndarray] = []
        k_list: list[np.ndarray] = []
        for sample_dir in sorted((OUTPUTS_ROOT / dataset).glob("sample_*")):
            srg_path = sample_dir / "srg" / f"{method}.npz"
            if not srg_path.is_file():
                continue
            data = np.load(srg_path)
            mif_list.append(data["mif_curve"])
            lif_list.append(data["lif_curve"])
            k_list.append(data["k_fractions"])
        if not mif_list:
            ax.set_visible(False)
            continue
        k = k_list[0]
        mif_mean = np.mean(np.stack(mif_list), axis=0)
        lif_mean = np.mean(np.stack(lif_list), axis=0)
        ax.plot(k, mif_mean, label="MIF", color="C0")
        ax.plot(k, lif_mean, label="LIF", color="C1")
        ax.set_title(method)
        ax.set_xlabel("fraction corrupted")
        ax.set_ylabel("p(target)")
        ax.legend(fontsize=8)

    for j in range(n_methods, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle(f"Mean MIF/LIF curves — {dataset}")
    fig.tight_layout()
    fig.savefig(OUTPUTS_ROOT / dataset / "srg_curves_grid.pdf", dpi=150)
    plt.close(fig)


def write_cross_dataset_plot(datasets: list[str]) -> Path:
    frames = [collect_srg_rows(d) for d in datasets]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out = OUTPUTS_ROOT / "cross_dataset_srg.png"
    if df.empty:
        return out

    summary = df.groupby(["dataset", "method"])["srg_score"].mean().reset_index()
    pivot = summary.pivot(index="method", columns="dataset", values="srg_score")
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean SRG")
    ax.set_title("Mean SRG by method and dataset")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    summary.to_csv(OUTPUTS_ROOT / "cross_dataset_srg.csv", index=False)
    return out
