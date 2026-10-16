"""Percent change of the budget checkpoints against the main multirun best pick."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from spec import (
    GRACOIMAGE_ROOT,
    GRACO_METHODS,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    METHOD_LABELS,
    PER_SAMPLE_COLUMNS,
    SPECS,
    comparison_root,
    load_config,
    metrics_root,
)

MEAN_COLUMNS = tuple(column for column in PER_SAMPLE_COLUMNS if column != "has_valid_cf")
UNSTRETCHED = GRACOIMAGE_ROOT / "results_bundle" / "cifar_unstretched_results"
MULTIRUN_JSON = GRACOIMAGE_ROOT / "multirun_classification.json"


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
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _percent(budget: float | None, baseline: float | None) -> float | None:
    if budget is None or baseline is None or not math.isfinite(baseline) or abs(baseline) < 1e-12:
        return None
    return (float(budget) - float(baseline)) / abs(float(baseline)) * 100.0


def direction_of(metric: str) -> str:
    if metric in HIGHER_IS_BETTER:
        return "higher"
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric.startswith("ssim") or metric == "validity":
        return "higher"
    return "lower"


def _per_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picked = []
    for row in rows:
        method = str(row.get("method", ""))
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


def _table_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            mean, std, n = _mean_std(
                [float(row[column]) for row in valid if row.get(column) is not None]
            )
            entry[f"{column}_mean"] = mean
            entry[f"{column}_std"] = std
            entry[f"{column}_n"] = n
        out.append(entry)
    return out


def _distribution(metrics_dir: Path) -> list[dict[str, Any]]:
    paired = {row["method"]: row for row in _read_csv(metrics_dir / "distribution_metrics.csv")}
    stratified = {row["method"]: row for row in _read_csv(metrics_dir / "cmmd_batch_stratified.csv")}
    fid = {row["method"]: row for row in _read_csv(metrics_dir / "batch_fid.csv")}
    rows = []
    for method in GRACO_METHODS:
        src: dict[str, Any] = {}
        src.update(paired.get(method, {}))
        src.update(stratified.get(method, {}))
        src.update(fid.get(method, {}))
        if not src:
            continue
        item: dict[str, Any] = {"method": method, "readout": METHOD_LABELS[method]}
        for key, value in src.items():
            if key in {"method", "note", "readout"}:
                continue
            number = _finite(value)
            item[key] = int(number) if key == "n_paired" and number is not None else number
        rows.append(item)
    return rows


def _baseline(dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    if dataset == "cifar":
        per_path = UNSTRETCHED / "unstretched_multirun_per_sample_metrics.csv"
        if not per_path.is_file():
            raise FileNotFoundError(f"Missing CIFAR unstretched baseline {per_path}")
        rows = _per_sample(_read_csv(per_path))
        means = _table_means(rows)
        distribution = _distribution_from_split(
            UNSTRETCHED / "unstretched_multirun_cmmd",
            UNSTRETCHED / "unstretched_multirun_batch_fid.csv",
        )
        return rows, means, distribution, str(per_path)
    if not MULTIRUN_JSON.is_file():
        raise FileNotFoundError(f"Missing multirun baseline {MULTIRUN_JSON}")
    payload = json.loads(MULTIRUN_JSON.read_text(encoding="utf-8"))
    block = payload["datasets"][SPECS[dataset].json_key]
    return (
        list(block.get("per_sample") or []),
        list(block.get("table_means") or []),
        list(block.get("distribution") or []),
        str(MULTIRUN_JSON),
    )


def _distribution_from_split(cmmd_dir: Path, batch_fid: Path) -> list[dict[str, Any]]:
    paired = {row["method"]: row for row in _read_csv(cmmd_dir / "distribution_metrics.csv")}
    stratified = {row["method"]: row for row in _read_csv(cmmd_dir / "cmmd_batch_stratified.csv")}
    fid = {row["method"]: row for row in _read_csv(batch_fid)}
    rows = []
    for method in GRACO_METHODS:
        src: dict[str, Any] = {}
        src.update(paired.get(method, {}))
        src.update(stratified.get(method, {}))
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


def _index_means(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["method"]): row for row in rows}


def _index_samples(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(row["sample_idx"]), str(row["method"])): row for row in rows}


def compare_dataset(dataset: str) -> Path:
    cfg = load_config(dataset)
    budget = cfg["budget"]
    population = int(budget["population_size"])
    main_evaluations = int(budget["main_population"]) * int(budget["main_logged_generations"])
    generations = [int(g) for g in budget["snapshot_gens"]]
    base_rows, base_means, base_dist, baseline_name = _baseline(dataset)
    base_mean_ix = _index_means(base_means)
    base_sample_ix = _index_samples(base_rows)
    base_dist_ix = _index_means(base_dist)

    flat: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    summary_gens: dict[str, Any] = {}

    for generation in generations:
        metrics = metrics_root(dataset, generation)
        per_path = metrics / "per_sample_metrics.csv"
        if not per_path.is_file():
            print(f"  no per-sample metrics yet for {dataset} generation {generation}", flush=True)
            continue
        rows = _per_sample(_read_csv(per_path))
        means = _table_means(rows)
        distribution = _distribution(metrics)
        n_evaluations = population * (generation + 1)
        fraction = n_evaluations / main_evaluations if main_evaluations else None
        summary_gens[str(generation)] = {
            "n_evaluations": n_evaluations,
            "main_evaluations": main_evaluations,
            "budget_fraction": fraction,
            "table_means": means,
            "distribution": distribution,
        }
        for row in means:
            method = str(row["method"])
            base = base_mean_ix.get(method, {})
            _append_mean_rows(
                flat,
                dataset=dataset,
                generation=generation,
                n_evaluations=n_evaluations,
                main_evaluations=main_evaluations,
                fraction=fraction,
                method=method,
                readout=str(row["readout"]),
                budget_row=row,
                baseline_row=base,
            )
        for row in distribution:
            method = str(row["method"])
            base = base_dist_ix.get(method, {})
            for key, value in row.items():
                if key in {"method", "readout"} or not isinstance(value, (int, float)):
                    continue
                if key in {"n", "n_valid", "n_paired"}:
                    continue
                base_value = _finite(base.get(key))
                budget_value = _finite(value)
                flat.append(
                    _change_row(
                        dataset=dataset,
                        generation=generation,
                        n_evaluations=n_evaluations,
                        main_evaluations=main_evaluations,
                        fraction=fraction,
                        method=method,
                        readout=METHOD_LABELS[method],
                        metric=key,
                        kind="distribution",
                        baseline=base_value,
                        budget=budget_value,
                    )
                )
        budget_ix = _index_samples(rows)
        for key, budget_row in budget_ix.items():
            base_row = base_sample_ix.get(key)
            if base_row is None:
                continue
            sample_idx, method = key
            for column in ("has_valid_cf", *MEAN_COLUMNS):
                paired.append(
                    {
                        "dataset": dataset,
                        "generation": generation,
                        "sample_idx": sample_idx,
                        "method": method,
                        "readout": METHOD_LABELS[method],
                        "metric": column,
                        "baseline": base_row.get(column),
                        "budget": budget_row.get(column),
                        "percent_change": _percent(
                            _finite(budget_row.get(column)),
                            _finite(base_row.get(column)),
                        ),
                    }
                )

    out_dir = comparison_root(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "percent_change.csv"
    _write_dicts(table_path, flat)
    _write_dicts(out_dir / "per_sample_paired.csv", paired)
    summary = {
        "dataset": dataset,
        "baseline": baseline_name,
        "population_size": population,
        "main_population": int(budget["main_population"]),
        "main_logged_generations": int(budget["main_logged_generations"]),
        "generations": summary_gens,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {table_path}", flush=True)
    return table_path


def _append_mean_rows(
    flat: list[dict[str, Any]],
    *,
    dataset: str,
    generation: int,
    n_evaluations: int,
    main_evaluations: int,
    fraction: float | None,
    method: str,
    readout: str,
    budget_row: dict[str, Any],
    baseline_row: dict[str, Any],
) -> None:
    metrics = ["validity", *MEAN_COLUMNS]
    for metric in metrics:
        if metric == "validity":
            budget_value = _finite(budget_row.get("validity"))
            base_value = _finite(baseline_row.get("validity"))
        else:
            budget_value = _finite(budget_row.get(f"{metric}_mean"))
            base_value = _finite(baseline_row.get(f"{metric}_mean"))
        flat.append(
            _change_row(
                dataset=dataset,
                generation=generation,
                n_evaluations=n_evaluations,
                main_evaluations=main_evaluations,
                fraction=fraction,
                method=method,
                readout=readout,
                metric=metric,
                kind="per_sample_mean",
                baseline=base_value,
                budget=budget_value,
            )
        )


def _change_row(
    *,
    dataset: str,
    generation: int,
    n_evaluations: int,
    main_evaluations: int,
    fraction: float | None,
    method: str,
    readout: str,
    metric: str,
    kind: str,
    baseline: float | None,
    budget: float | None,
) -> dict[str, Any]:
    point_diff = None
    if metric == "validity" and baseline is not None and budget is not None:
        point_diff = (budget - baseline) * 100.0
    return {
        "dataset": dataset,
        "generation": generation,
        "n_evaluations": n_evaluations,
        "main_evaluations": main_evaluations,
        "budget_fraction": fraction,
        "method": method,
        "readout": readout,
        "metric": metric,
        "kind": kind,
        "direction": direction_of(metric),
        "baseline": baseline,
        "budget": budget,
        "percent_change": _percent(budget, baseline),
        "validity_percentage_points": point_diff,
    }


def _write_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
