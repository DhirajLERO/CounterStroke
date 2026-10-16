#!/usr/bin/env python3
"""Export a static site bundle from experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_PKG = Path(__file__).resolve().parent
_GRACO = _PKG.parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from afp.path_setup import setup_import_paths

setup_import_paths()

from afp.metrics_catalog import (  # noqa: E402
    METRIC_GROUPS,
    METRICS,
    METHOD_ORDER,
)
from afp.metrics_preprocess import prepare_per_sample_df  # noqa: E402
from afp.paths import GRACOIMAGE_ROOT  # noqa: E402

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc

BUNDLE_VERSION = 1
HEAT_METHODS = (
    "sum_intensity",
    "mean_intensity",
    "change_frequency",
    "max_intensity",
    "cochange_strict",
    "spatial_coherence",
)
ATTR_METHODS = (
    "integrated_gradients",
    "input_x_gradient",
    "smoothgrad",
    "gradcam",
)
PUBLIC_METHOD_LABELS = {
    "counterfactual_pt": "GradCF",
    "cfproto": "CFProto",
    "cfrl": "CFRL",
    "leapfactual": "Leap",
    "leapfactual_r": "Leap-R",
    "graco_base_o2": "CS_sparse",
    "graco_base_o3": "CS_prox",
    "graco50": "CS_50",
}
SECONDARY_METHOD_LABELS = {
    "counterfactual_pt": "Wachter",
    "graco_base_o2": "Graco O2",
    "graco_base_o3": "Graco O3",
    "graco50": "Graco50",
}
HEAT_LABELS = {
    "sum_intensity": "Sum intensity",
    "mean_intensity": "Mean intensity",
    "change_frequency": "Change frequency",
    "max_intensity": "Max intensity",
    "cochange_strict": "Co-change",
    "spatial_coherence": "Spatial coherence",
}
ATTR_LABELS = {
    "integrated_gradients": "Integrated Gradients",
    "input_x_gradient": "Input × Gradient",
    "smoothgrad": "SmoothGrad",
    "gradcam": "Grad-CAM",
}
CIFAR_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
MNIST_NAMES = [str(i) for i in range(10)]
DATASET_KEYS = ("mnist", "cifar10", "imagenet100")
BUNDLE_CSV_ROOT = GRACOIMAGE_ROOT / "results_bundle" / "raw" / "generation"
BUNDLE_META = {
    "mnist": GRACOIMAGE_ROOT / "results_bundle" / "raw" / "generation_meta" / "MNIST" / "outputs" / "generation",
    "cifar10": GRACOIMAGE_ROOT / "results_bundle" / "raw" / "generation_meta" / "Cifar10" / "outputs" / "generation",
    "imagenet100": GRACOIMAGE_ROOT / "results_bundle" / "raw" / "generation_meta" / "Imagenet" / "outputs" / "generation",
}
BUNDLE_CSV = {
    "mnist": BUNDLE_CSV_ROOT / "MNIST",
    "cifar10": BUNDLE_CSV_ROOT / "Cifar10",
    "imagenet100": BUNDLE_CSV_ROOT / "Imagenet",
}

PAPER = {
    "title": "CounterStroke: Investigating Evolutionary Image Counterfactual Generation",
    "authors": [
        {"name": "Dhiraj Kumar Singh", "email": "dhirajkumar.singh@ul.ie"},
        {"name": "Conor Ryan", "email": "conor.ryan@ul.ie"},
    ],
    "affiliation": "Biocomputing & Developmental Systems (BDS) group, Department of Computer Science and Information Systems, University of Limerick, Ireland",
    "keywords": [
        "counterfactual explanations",
        "evolutionary computation",
        "explainable AI",
        "image classification",
        "multi-objective optimisation",
        "feature attribution",
    ],
    "abstract": (
        "Counterfactual explanations help interpret image classifiers by asking what changes "
        "would alter a prediction. Existing methods typically rely on gradient-based pixel-space "
        "optimisation or generative modelling. We introduce CounterStroke, an evolutionary "
        "framework that evolves structured drawing programs over the image. Evaluated on MNIST, "
        "CIFAR-10, and ImageNet-100, CounterStroke achieves 100% validity on all classification "
        "datasets (vs 81% GradCF and 52% CFProto on CIFAR), with substantially sparser edits. "
        "Aggregating successful phenotypes yields attribution maps competitive under SRG "
        "(MNIST 0.483, CIFAR 0.224). The same representation supports blank-canvas synthesis "
        "of classifier-sufficient evidence with no factual image."
    ),
    "highlights": [
        {"id": "validity", "label": "Validity", "value": "100%", "detail": "MNIST, CIFAR-10, ImageNet-100"},
        {"id": "cifar_baselines", "label": "CIFAR baselines", "value": "81% / 52%", "detail": "GradCF / CFProto validity"},
        {"id": "sparsity", "label": "Sparsity effect", "value": "r = −0.84", "detail": "MNIST vs denser baselines"},
        {"id": "srg", "label": "SRG (MNIST)", "value": "0.483", "detail": "Phenotype attribution vs XAI"},
    ],
    "contributions": [
        "Grammar-guided evolutionary sketch programs that produce valid, sparse, inspectable image counterfactuals.",
        "Benchmarks on MNIST, CIFAR-10, and ImageNet-100 against gradient, prototype, and generative methods, plus Deep-SCM (MorphoMNIST, CelebA).",
        "Six phenotype-based attribution maps competitive under SRG on MNIST and CIFAR.",
        "Blank-canvas synthesis of classifier-sufficient sketches from an empty image.",
    ],
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _as_hwc_uint8(img_01: np.ndarray) -> np.ndarray:
    d = np.asarray(img_01, dtype=np.float32)
    d = np.clip(d, 0.0, 1.0)
    if d.ndim == 2:
        g = (d * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    if d.ndim == 3 and d.shape[-1] == 1:
        g = (d[:, :, 0] * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    if d.ndim == 3 and d.shape[0] == 3 and d.shape[-1] != 3:
        d = np.transpose(d, (1, 2, 0))
    return (np.clip(d, 0, 1) * 255).astype(np.uint8)


def _save_webp(path: Path, img_01: np.ndarray, *, quality: int, thumb: int | None = None) -> str:
    rgb = _as_hwc_uint8(img_01)
    im = Image.fromarray(rgb, mode="RGB")
    if thumb:
        im = im.resize((thumb, thumb), Image.Resampling.NEAREST if max(im.size) <= 64 else Image.Resampling.BILINEAR)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        im.save(path, format="WEBP", quality=int(quality), method=4)
        return path.name
    except Exception:
        png = path.with_suffix(".png")
        im.save(png, format="PNG")
        return png.name


def _save_heat_png(path: Path, heat: np.ndarray) -> None:
    h = np.asarray(heat, dtype=np.float32)
    h = np.clip(h, 0.0, 1.0)
    if h.ndim == 3:
        h = h.max(axis=-1)
    img = Image.fromarray((h * 255).astype(np.uint8), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def _dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.1f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


def class_names_for(dataset: str) -> list[str]:
    if dataset == "mnist":
        return list(MNIST_NAMES)
    if dataset == "cifar10":
        return list(CIFAR_NAMES)
    try:
        from afp.class_names import get_class_names

        return list(get_class_names(dataset, None))
    except Exception:
        return [f"class_{i}" for i in range(100)]


def catalog_payload() -> dict[str, Any]:
    return {
        "methods": [
            {
                "id": m,
                "label": PUBLIC_METHOD_LABELS.get(m, m),
                "secondary": SECONDARY_METHOD_LABELS.get(m),
                "family": "counterstroke" if m.startswith("graco") else "baseline",
            }
            for m in METHOD_ORDER
        ],
        "metricGroups": list(METRIC_GROUPS),
        "metrics": [
            {
                "id": m.id,
                "label": m.label,
                "group": m.group,
                "description": m.description,
                "direction": m.direction,
                "csvSource": m.csv_source,
            }
            for m in METRICS
        ],
        "heats": [{"id": k, "label": HEAT_LABELS[k]} for k in HEAT_METHODS],
        "attribution": [{"id": k, "label": ATTR_LABELS[k]} for k in ATTR_METHODS],
    }


def _load_csv(path: Path):
    import pandas as pd

    if not path.is_file():
        return None
    return pd.read_csv(path)


def _filter_methods(df, methods: list[str]):
    if df is None or not methods or "method" not in df.columns:
        return df
    return df[df["method"].isin(methods)]


def _aggregate_by_method(df, metric, methods: list[str]):
    import pandas as pd

    sub = _filter_methods(df, methods)
    if sub is None or sub.empty:
        return pd.DataFrame()
    if metric.id == "success_class_changed_rate":
        if "success_plot" in sub.columns:
            g = sub.groupby("method", sort=False)["success_plot"].agg(["mean", "std", "count"])
        elif "pred_class_cf" in sub.columns and "y_true" in sub.columns:
            sub = sub.assign(_ok=sub["pred_class_cf"] != sub["y_true"])
            g = sub.groupby("method", sort=False)["_ok"].agg(["mean", "std", "count"])
        else:
            return pd.DataFrame()
        return g.reset_index()
    col = metric.csv_column
    if not col or col not in sub.columns:
        return pd.DataFrame()
    return sub.groupby("method", sort=False)[col].agg(["mean", "std", "count"]).reset_index()


def _distribution_table(df, col: str, methods: list[str]):
    import pandas as pd

    if df is None or col not in df.columns:
        return pd.DataFrame()
    sub = _filter_methods(df, methods)
    cols = ["method", col]
    if "n_paired" in sub.columns:
        cols.append("n_paired")
    return sub[cols]


def _summary_rows_for_metric(metric, summary_df, methods: list[str]):
    import pandas as pd

    if summary_df is None or summary_df.empty or "metric" not in summary_df.columns:
        return pd.DataFrame()
    key = metric.csv_column or metric.id
    rows = summary_df[summary_df["metric"] == key]
    if metric.id == "success_class_changed_rate":
        rows = summary_df[summary_df["metric"] == "success_class_changed_rate"]
    if methods and "method" in rows.columns:
        rows = rows[rows["method"].isin(methods)]
    return rows


def _metric_table(metric, frames: dict, df_valid, methods: list[str]):
    import pandas as pd

    if metric.csv_source == "per_sample" and df_valid is not None:
        return _aggregate_by_method(df_valid, metric, methods=methods)
    if metric.csv_source == "distribution" and metric.csv_column:
        return _distribution_table(frames.get("distribution"), metric.csv_column, methods)
    if metric.csv_source == "batch_fid" and metric.csv_column:
        return _distribution_table(frames.get("batch_fid"), metric.csv_column, methods)
    if metric.csv_source == "cmmd_batch" and metric.csv_column:
        return _distribution_table(frames.get("cmmd_batch"), metric.csv_column, methods)
    if metric.csv_source == "cmmd_per_sample" and metric.csv_column:
        df = frames.get("cmmd_per_sample")
        if df is not None and "has_valid_cf" in df.columns:
            df = df.loc[df["has_valid_cf"] == 1]
        if df is not None and metric.csv_column in df.columns:
            sub = df[df["method"].isin(methods)] if methods else df
            g = sub.groupby("method", sort=False)[metric.csv_column].agg(["mean", "std", "count"])
            return g.reset_index()
    summary = _summary_rows_for_metric(metric, frames.get("summary"), methods)
    if summary is not None and not summary.empty:
        return summary
    return pd.DataFrame()


def _rows_mean_err(table, metric, methods: list[str], error_bars: str = "sem") -> dict[str, dict[str, float]]:
    import pandas as pd

    out: dict[str, dict[str, float]] = {}
    if table is None or table.empty or "method" not in table.columns:
        return out
    for method in methods:
        sub = table.loc[table["method"] == method]
        if sub.empty:
            continue
        row = sub.iloc[0]
        if "mean" in table.columns:
            mean = float(row["mean"])
            std = float(row["std"]) if "std" in table.columns and pd.notna(row.get("std")) else 0.0
            count = float(row["count"]) if "count" in table.columns and pd.notna(row.get("count")) else 0.0
            if error_bars == "none":
                err = 0.0
            elif error_bars == "std":
                err = std if count > 1 else 0.0
            else:
                err = (std / (count**0.5)) if count > 1 else 0.0
            out[method] = {"mean": mean, "err": err, "n": count}
        else:
            col = metric.csv_column
            if col and col in table.columns and pd.notna(row.get(col)):
                mean = float(row[col])
            elif "value" in table.columns and pd.notna(row.get("value")):
                mean = float(row["value"])
            else:
                continue
            n = float(row["n"]) if "n" in table.columns and pd.notna(row.get("n")) else 0.0
            out[method] = {"mean": mean, "err": 0.0, "n": n}
    return out


def resolve_metrics_csv_dir(dataset: str) -> Path | None:
    from afp.datasets import get_app_dataset_spec

    spec = get_app_dataset_spec(dataset)
    if (spec.default_gen_root / "per_sample_metrics.csv").is_file():
        return spec.default_gen_root
    fallback = BUNDLE_CSV.get(dataset)
    if fallback is not None and (fallback / "per_sample_metrics.csv").is_file():
        return fallback
    return None


def resolve_metrics_meta_dir(dataset: str, csv_dir: Path) -> Path:
    if any(csv_dir.glob("sample_*/meta.json")):
        return csv_dir
    fallback = BUNDLE_META.get(dataset)
    if fallback is not None and fallback.is_dir():
        return fallback
    return csv_dir


def export_metrics(out_root: Path, datasets: list[str]) -> dict[str, bool]:
    present: dict[str, bool] = {}
    scatter_dir = out_root / "metrics" / "scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        csv_dir = resolve_metrics_csv_dir(ds)
        if csv_dir is None:
            present[ds] = False
            print(f"  [metrics] {ds}: no CSVs found")
            continue
        meta_dir = resolve_metrics_meta_dir(ds, csv_dir)
        summary = _load_csv(csv_dir / "summary_tables.csv")
        if summary is None:
            summary = _load_csv(csv_dir.parent / "results" / "generation" / "summary_tables.csv")
        frames = {
            "per_sample": _load_csv(csv_dir / "per_sample_metrics.csv"),
            "distribution": _load_csv(csv_dir / "distribution_metrics.csv"),
            "batch_fid": _load_csv(csv_dir / "batch_fid.csv"),
            "cmmd_per_sample": _load_csv(csv_dir / "cmmd_per_sample.csv"),
            "cmmd_batch": _load_csv(csv_dir / "cmmd_batch_stratified.csv"),
            "summary": summary,
        }
        df_valid = None
        ps = frames["per_sample"]
        if ps is not None and not ps.empty:
            df_valid = prepare_per_sample_df(ps, meta_dir)
        methods = list(METHOD_ORDER)
        payload: dict[str, Any] = {
            "dataset": ds,
            "source": str(csv_dir),
            "methods": methods,
            "metrics": {},
        }
        for metric in METRICS:
            table = _metric_table(metric, frames, df_valid, methods)
            payload["metrics"][metric.id] = {
                "id": metric.id,
                "label": metric.label,
                "group": metric.group,
                "direction": metric.direction,
                "byMethod": _rows_mean_err(table, metric, methods),
            }
        _write_json(out_root / "metrics" / f"{ds}.json", payload)

        scatter_rows: list[dict[str, Any]] = []
        if df_valid is not None:
            cols = [
                c
                for c in (
                    "sample_idx",
                    "y_true",
                    "y_intended_target",
                    "method",
                    "l2_distance",
                    "sparsity_count",
                    "sparsity_frac",
                    "success_plot",
                    "pred_class_cf",
                    "p_pred_cf",
                )
                if c in df_valid.columns
            ]
            for rec in df_valid[cols].to_dict(orient="records"):
                row = {k: (None if _is_nan(v) else _json_num(v)) for k, v in rec.items()}
                scatter_rows.append(row)
        _write_json(scatter_dir / f"{ds}.json", {"dataset": ds, "rows": scatter_rows})
        present[ds] = True
        print(f"  [metrics] {ds}: {len(payload['metrics'])} metrics, {len(scatter_rows)} scatter rows")
    return present


def _is_nan(v: Any) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except Exception:
        return False


def _json_num(v: Any) -> Any:
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return v


def _o1_hist(values: list[float], bins: int = 20) -> dict[str, list[float]]:
    if not values:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=bins)
    return {"edges": [float(x) for x in edges], "counts": [int(x) for x in counts]}


def _pick_gallery(records, n: int) -> list:
    if not records:
        return []
    by_cls: dict[int, list] = {}
    for r in records:
        by_cls.setdefault(int(r.pred_class), []).append(r)
    for lst in by_cls.values():
        lst.sort(key=lambda x: float(x.pred_prob), reverse=True)
    picked = []
    classes = sorted(by_cls, key=lambda c: -len(by_cls[c]))
    i = 0
    while len(picked) < n and any(by_cls[c] for c in classes):
        c = classes[i % len(classes)]
        if by_cls[c]:
            picked.append(by_cls[c].pop(0))
        i += 1
    return picked[:n]


def _softmax_safe(spec, img_alibi, weights: Path, device) -> list[float] | None:
    if weights is None or not Path(weights).is_file():
        return None
    try:
        from afp.classifier import predict_probs

        probs = predict_probs(spec, img_alibi, Path(weights), device)
        return [float(x) for x in np.asarray(probs).ravel()]
    except Exception as exc:
        print(f"    softmax failed: {exc}")
        return None


def _display_from_alibi(alibi: np.ndarray) -> np.ndarray:
    from afp.inputs import alibi_to_display_01

    return alibi_to_display_01(alibi)


def harvest_factual_sample(
    *,
    spec,
    entry,
    names: list[str],
    out_dir: Path,
    args: argparse.Namespace,
    missing: list[dict[str, Any]],
    device,
) -> dict[str, Any] | None:
    from afp.analysis import (
        analyse_factual_distribution,
        collect_diffs_for_target,
        heat_from_diffs,
    )
    from afp.generation_cfs import load_method_cfs, load_sample_meta
    from afp.graco_paths import find_graco_sample_dir
    from afp.inputs import load_factual_alibi
    from afp.population_io import load_final_population
    from afp.render import record_to_alibi

    sample_idx = int(entry.sample_idx)
    dest = out_dir / str(sample_idx)
    dest.mkdir(parents=True, exist_ok=True)

    try:
        alibi = load_factual_alibi(spec, entry, spec.default_data_root)
        disp = _display_from_alibi(alibi)
    except Exception as exc:
        missing.append({"dataset": spec.name, "kind": "factual", "sample_idx": sample_idx, "error": str(exc)})
        return None

    if args.dry_run:
        return {"sample_idx": sample_idx, "y_true": int(entry.y_true)}

    _save_webp(dest / "input.webp", disp, quality=args.webp_quality)
    _save_webp(dest / "thumb.webp", disp, quality=max(50, args.webp_quality - 10), thumb=96)

    sample_dir = find_graco_sample_dir(spec.default_factual_graco_root, sample_idx, int(entry.y_true))
    records, src = ([], "none")
    if sample_dir is not None:
        records, src = load_final_population(sample_dir, run_id=args.run_id)

    result = analyse_factual_distribution(records, prob_threshold=args.prob_threshold)
    meta_gen = load_sample_meta(spec.default_gen_root / _sample_dir_name(spec.name, sample_idx))
    y_target = meta_gen.get("y_intended_target")
    if y_target is None:
        y_target = meta_gen.get("y_target_shift")

    method_imgs = load_method_cfs(spec, sample_idx, gen_root=spec.default_gen_root)
    methods_meta: dict[str, Any] = {}
    for method, img in method_imgs.items():
        if img is None:
            continue
        _save_webp(dest / "methods" / f"{method}.webp", img, quality=args.webp_quality)
        entry_m: dict[str, Any] = {"file": f"methods/{method}.webp"}
        gsr = (meta_gen.get("graco_selected_row") or {}).get(method) or {}
        if isinstance(gsr, dict) and gsr:
            entry_m["pred_class"] = gsr.get("pred_class")
            entry_m["pred_prob"] = gsr.get("pred_prob")
            entry_m["o1"] = gsr.get("o1")
            entry_m["o2"] = gsr.get("o2")
            entry_m["o3"] = gsr.get("o3")
        if args.softmax and spec.default_weights.is_file():
            from afp.classifier import cf_01_to_alibi

            try:
                probs = _softmax_safe(spec, cf_01_to_alibi(spec, img), spec.default_weights, device)
                if probs:
                    entry_m["probs"] = probs
                    entry_m["pred_class"] = int(np.argmax(probs))
                    entry_m["pred_prob"] = float(max(probs))
            except Exception:
                pass
        methods_meta[method] = entry_m

    gallery = _pick_gallery(result.filtered, args.pop_gallery)
    pop_meta = []
    for i, rec in enumerate(gallery):
        try:
            cf_alibi = record_to_alibi(spec, rec, alibi)
            cf_disp = _display_from_alibi(cf_alibi)
            fname = f"{i:02d}.webp"
            _save_webp(dest / "population" / fname, cf_disp, quality=args.webp_quality)
        except Exception:
            fname = ""
        pop_meta.append(
            {
                "id": i,
                "file": f"population/{fname}" if fname else None,
                "pred_class": int(rec.pred_class),
                "pred_prob": float(rec.pred_prob),
                "o1": None if rec.o1 is None else float(rec.o1),
                "o2": None if rec.o2 is None else float(rec.o2),
                "o3": None if rec.o3 is None else float(rec.o3),
                "phenotype": (rec.phenotype or "")[:2000],
            }
        )

    heats_written: list[str] = []
    if result.top_class is not None and result.filtered:
        diffs, n_used, shape = collect_diffs_for_target(
            spec,
            alibi,
            result.filtered,
            target_pred_class=int(result.top_class),
            max_render=args.max_renders,
        )
        if diffs:
            for hm in HEAT_METHODS:
                heat = heat_from_diffs(diffs, shape, hm, cochange_tau=0.5)
                _save_heat_png(dest / "heats" / f"{hm}.png", heat)
                heats_written.append(hm)
        _ = n_used

    attr_written: list[str] = []
    if not args.skip_attribution:
        from afp.attribution_maps import ensure_baseline_map

        target = int(result.top_class) if result.top_class is not None else int(entry.y_true)
        for am in ATTR_METHODS:
            try:
                res = ensure_baseline_map(
                    dataset=spec.name,
                    sample_idx=sample_idx,
                    method=am,
                    img_alibi=alibi,
                    target_class=target,
                    weights=spec.default_weights,
                    device=device,
                    cache=True,
                )
                if res.importance is None:
                    continue
                imp = np.asarray(res.importance, dtype=np.float32)
                if imp.ndim == 3:
                    imp = np.abs(imp).max(axis=0 if imp.shape[0] in (1, 3) else -1)
                mx = float(imp.max()) if imp.size else 0.0
                if mx > 0:
                    imp = imp / mx
                _save_heat_png(dest / "attr" / f"{am}.png", imp)
                attr_written.append(am)
            except Exception as exc:
                missing.append(
                    {
                        "dataset": spec.name,
                        "kind": "attr",
                        "sample_idx": sample_idx,
                        "method": am,
                        "error": str(exc),
                    }
                )

    input_probs = None
    if args.softmax:
        input_probs = _softmax_safe(spec, alibi, spec.default_weights, device)

    o1_vals = [float(r.o1) for r in records if r.o1 is not None]
    y_true = int(entry.y_true)
    payload = {
        "sample_idx": sample_idx,
        "y_true": y_true,
        "y_true_name": names[y_true] if 0 <= y_true < len(names) else str(y_true),
        "y_intended_target": None if y_target is None else int(y_target),
        "n_population": result.n_total,
        "n_filtered": result.n_filtered,
        "top_class": result.top_class,
        "source": src,
        "distribution": {str(k): int(v) for k, v in result.distribution.items()},
        "o1_hist": _o1_hist(o1_vals),
        "input_probs": input_probs,
        "methods": methods_meta,
        "population": pop_meta,
        "heats": heats_written,
        "attr": attr_written,
    }
    _write_json(dest / "meta.json", payload)
    return {
        "sample_idx": sample_idx,
        "y_true": y_true,
        "y_true_name": payload["y_true_name"],
        "y_intended_target": payload["y_intended_target"],
        "n_filtered": result.n_filtered,
        "top_class": result.top_class,
        "thumb": f"{sample_idx}/thumb.webp",
    }


def _sample_dir_name(dataset: str, sample_idx: int) -> str:
    if dataset == "imagenet100":
        return f"sample_{int(sample_idx):08d}"
    return f"sample_{int(sample_idx):05d}"


def harvest_blank_target(
    *,
    spec,
    k: int,
    names: list[str],
    out_dir: Path,
    args: argparse.Namespace,
    missing: list[dict[str, Any]],
    device,
    ssim_pool: dict[int, list[np.ndarray]] | None,
) -> dict[str, Any]:
    from afp.analysis import collect_diffs_for_target, heat_from_diffs
    from afp.blank_belief import load_target_drilldown
    from afp.blank_canvas import blank_alibi_for_dataset
    from afp.blank_prob_bins import pick_records_by_prob_bins
    from afp.render import record_to_alibi

    dest = out_dir / f"target_{k:03d}"
    dest.mkdir(parents=True, exist_ok=True)
    drill = load_target_drilldown(spec.default_blank_graco_root, k, run_id=args.run_id)
    row = {
        "wanted_class": k,
        "wanted_name": names[k] if 0 <= k < len(names) else str(k),
        "n_total": drill.n_total,
        "success_rate": drill.success_rate,
        "mean_p_blank": drill.mean_p_blank,
        "mode_pred": drill.mode_pred,
        "has_csv": bool(drill.records),
    }
    if args.dry_run:
        return row

    blank = blank_alibi_for_dataset(spec.name)
    blank_disp = _display_from_alibi(blank)
    _save_webp(dest / "blank.webp", blank_disp, quality=args.webp_quality)

    bins = pick_records_by_prob_bins(drill.records, k)
    bin_meta = []
    best_rec = None
    if drill.filtered:
        best_rec = max(drill.filtered, key=lambda r: float(r.pred_prob))
        if best_rec.pred_class != k:
            hits = [r for r in drill.records if r.pred_class == k]
            if hits:
                best_rec = max(hits, key=lambda r: float(r.pred_prob))

    if best_rec is not None:
        try:
            cf = record_to_alibi(spec, best_rec, blank)
            _save_webp(dest / "best.webp", _display_from_alibi(cf), quality=args.webp_quality)
        except Exception as exc:
            missing.append({"dataset": spec.name, "kind": "blank_best", "class": k, "error": str(exc)})

    for bp in bins:
        item: dict[str, Any] = {"lo": bp.lo, "hi": bp.hi, "label": bp.label, "file": None, "ssim": None}
        if bp.record is None:
            bin_meta.append(item)
            continue
        stem = f"{int(bp.lo * 100)}_{int(bp.hi * 100)}"
        try:
            cf = record_to_alibi(spec, bp.record, blank)
            disp = _display_from_alibi(cf)
            _save_webp(dest / "bins" / f"{stem}.webp", disp, quality=args.webp_quality)
            item["file"] = f"bins/{stem}.webp"
            item["pred_prob"] = float(bp.record.pred_prob)
            item["phenotype"] = (bp.record.phenotype or "")[:2000]
            if ssim_pool and k in ssim_pool and not args.skip_ssim:
                from afp.blank_ssim_refs import best_ssim_ref

                ref, score = best_ssim_ref(disp, ssim_pool[k], device=str(device))
                if ref is not None:
                    _save_webp(dest / "ssim" / f"{stem}.webp", ref, quality=args.webp_quality)
                    item["ssim"] = {"file": f"ssim/{stem}.webp", "score": score}
        except Exception as exc:
            missing.append({"dataset": spec.name, "kind": "blank_bin", "class": k, "error": str(exc)})
        bin_meta.append(item)

    heats: list[str] = []
    if drill.filtered:
        target = k if any(r.pred_class == k for r in drill.filtered) else int(drill.filtered[0].pred_class)
        diffs, _, shape = collect_diffs_for_target(
            spec,
            blank,
            drill.filtered,
            target_pred_class=target,
            max_render=args.max_renders,
        )
        if diffs:
            for hm in HEAT_METHODS:
                heat = heat_from_diffs(diffs, shape, hm, cochange_tau=0.5)
                _save_heat_png(dest / "heats" / f"{hm}.png", heat)
                heats.append(hm)

    blank_probs = _softmax_safe(spec, blank, spec.default_weights, device) if args.softmax else None
    payload = {
        **row,
        "distribution": {str(a): int(b) for a, b in drill.dist_full.items()},
        "distribution_filtered": {str(a): int(b) for a, b in drill.dist_filtered.items()},
        "o1_hist": _o1_hist(drill.o1_values),
        "bins": bin_meta,
        "heats": heats,
        "blank_probs": blank_probs,
    }
    _write_json(dest / "meta.json", payload)
    return row


def _synthetic_image(dataset: str, label: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if dataset == "mnist":
        img = np.zeros((28, 28), dtype=np.float32)
        cy, cx = 14, 14
        for y in range(28):
            for x in range(28):
                r = math.hypot(y - cy, x - cx)
                if 6 + (label % 5) < r < 10 + (label % 4):
                    img[y, x] = 0.85
        img += rng.normal(0, 0.04, img.shape).astype(np.float32)
        return np.clip(img, 0, 1)
    if dataset == "cifar10":
        img = rng.random((32, 32, 3), dtype=np.float32) * 0.25
        img[:, :, label % 3] += 0.45
        return np.clip(img, 0, 1)
    img = rng.random((96, 96, 3), dtype=np.float32) * 0.3
    img[20:76, 20:76, :] = (0.2, 0.45, 0.35)
    return np.clip(img, 0, 1)


def write_fixture(out_root: Path, quality: int) -> None:
    names = {
        "mnist": MNIST_NAMES,
        "cifar10": CIFAR_NAMES,
        "imagenet100": [f"class_{i}" for i in range(100)],
    }
    for ds, size_n in (("mnist", 28), ("cifar10", 32), ("imagenet100", 96)):
        idx_rows = []
        for i, y in enumerate((3, 7) if ds != "imagenet100" else (0, 1)):
            sid = 1000 + i
            dest = out_root / "factual" / ds / str(sid)
            dest.mkdir(parents=True, exist_ok=True)
            src = _synthetic_image(ds, y, seed=10 + i)
            cf = np.clip(src + 0.15 * _synthetic_image(ds, (y + 1) % 10, seed=40 + i), 0, 1)
            _save_webp(dest / "input.webp", src, quality=quality)
            _save_webp(dest / "thumb.webp", src, quality=quality, thumb=96)
            for m in METHOD_ORDER:
                jitter = cf if m.startswith("graco") else np.clip(src * 0.7 + 0.3 * cf, 0, 1)
                _save_webp(dest / "methods" / f"{m}.webp", jitter, quality=quality)
            for p in range(4):
                _save_webp(dest / "population" / f"{p:02d}.webp", np.clip(cf + 0.05 * p, 0, 1), quality=quality)
            heat = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
            heat[src.shape[0] // 4 : 3 * src.shape[0] // 4, src.shape[1] // 3 : 2 * src.shape[1] // 3] = 1.0
            for hm in HEAT_METHODS:
                _save_heat_png(dest / "heats" / f"{hm}.png", heat)
            for am in ATTR_METHODS:
                _save_heat_png(dest / "attr" / f"{am}.png", heat * 0.7)
            n_cls = 10 if ds != "imagenet100" else 100
            meta = {
                "sample_idx": sid,
                "y_true": y,
                "y_true_name": names[ds][y],
                "y_intended_target": (y + 1) % min(n_cls, 10),
                "n_population": 200,
                "n_filtered": 48,
                "top_class": (y + 1) % min(n_cls, 10),
                "source": "fixture",
                "distribution": {str((y + 1) % min(n_cls, 10)): 40, str(y): 8},
                "o1_hist": {"edges": [-1, -0.5, 0, 0.5], "counts": [20, 18, 10]},
                "methods": {
                    m: {"file": f"methods/{m}.webp", "pred_class": (y + 1) % min(n_cls, 10), "pred_prob": 0.7}
                    for m in METHOD_ORDER
                },
                "population": [
                    {
                        "id": p,
                        "file": f"population/{p:02d}.webp",
                        "pred_class": (y + 1) % min(n_cls, 10),
                        "pred_prob": 0.9 - 0.05 * p,
                        "o1": -0.4,
                        "phenotype": f"(stroke fixture {p})",
                    }
                    for p in range(4)
                ],
                "heats": list(HEAT_METHODS),
                "attr": list(ATTR_METHODS),
                "fixture": True,
            }
            _write_json(dest / "meta.json", meta)
            idx_rows.append(
                {
                    "sample_idx": sid,
                    "y_true": y,
                    "y_true_name": names[ds][y],
                    "y_intended_target": (y + 1) % min(n_cls, 10),
                    "n_filtered": 48,
                    "top_class": (y + 1) % min(n_cls, 10),
                    "thumb": f"{sid}/thumb.webp",
                    "fixture": True,
                }
            )
        _write_json(out_root / "factual" / ds / "index.json", {"dataset": ds, "samples": idx_rows})

        blank_rows = []
        for k in (0, 1) if ds != "imagenet100" else (0, 1):
            dest = out_root / "blank" / ds / f"target_{k:03d}"
            dest.mkdir(parents=True, exist_ok=True)
            blank = np.zeros_like(_synthetic_image(ds, k, seed=1))
            if ds != "mnist":
                blank = np.ones_like(_synthetic_image(ds, k, seed=1)) * 0.95
            evid = _synthetic_image(ds, k, seed=99)
            _save_webp(dest / "blank.webp", blank, quality=quality)
            _save_webp(dest / "best.webp", evid, quality=quality)
            heat = evid if evid.ndim == 2 else evid.mean(axis=-1)
            for hm in HEAT_METHODS:
                _save_heat_png(dest / "heats" / f"{hm}.png", heat)
            bins = []
            for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)):
                stem = f"{int(lo * 100)}_{int(hi * 100)}"
                _save_webp(dest / "bins" / f"{stem}.webp", evid * (lo + 0.2), quality=quality)
                bins.append({"lo": lo, "hi": hi, "label": f"{int(lo*100)}–{int(hi*100)}%", "file": f"bins/{stem}.webp"})
            row = {
                "wanted_class": k,
                "wanted_name": names[ds][k],
                "n_total": 200,
                "success_rate": 0.62,
                "mean_p_blank": 0.11,
                "mode_pred": k,
                "has_csv": True,
                "fixture": True,
            }
            _write_json(
                dest / "meta.json",
                {**row, "bins": bins, "heats": list(HEAT_METHODS), "distribution": {str(k): 120, "1": 80}},
            )
            blank_rows.append(row)
        _write_json(out_root / "blank" / ds / "index.json", {"dataset": ds, "targets": blank_rows})
    print("  [fixture] wrote synthetic factual + blank samples")


def zip_bundle(src: Path, zip_path: Path, *, arc_from: Path | None = None) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    root = arc_from or src
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(root))
    print(f"Wrote zip {zip_path} ({_fmt_bytes(zip_path.stat().st_size)})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harvest static CounterStroke Pages data")
    p.add_argument("--out", type=Path, default=_GRACO / "_site_export")
    p.add_argument("--zip", type=Path, default=None)
    p.add_argument("--datasets", type=str, default="mnist,cifar10,imagenet100")
    p.add_argument("--pop-gallery", type=int, default=12)
    p.add_argument("--max-renders", type=int, default=200)
    p.add_argument("--webp-quality", type=int, default=82)
    p.add_argument("--heat-png", action="store_true", help="Write 8-bit heat PNGs (always on; kept for CLI compatibility)")
    p.add_argument("--prob-threshold", type=float, default=0.5)
    p.add_argument("--run-id", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--metrics-only", action="store_true", help="Export metrics JSON only (no images)")
    p.add_argument(
        "--evolution-only",
        action="store_true",
        help="Export only 30-run evolution curves + Pareto JSON into --out/evolution/ (does not redo factual/metrics)",
    )
    p.add_argument(
        "--skip-evolution",
        action="store_true",
        help="Skip 30-run evolution harvest during a full/metrics run",
    )
    p.add_argument(
        "--evolution-root",
        type=Path,
        default=None,
        help="Folder containing results/{mnist,cifar,imagenet}/ (default: GracoImage/evolution_experiment)",
    )
    p.add_argument("--write-fixture", action="store_true", help="Write synthetic explorer and/or evolution JSON when real traces are missing")
    p.add_argument("--skip-blank", action="store_true")
    p.add_argument("--skip-factual", action="store_true")
    p.add_argument("--skip-attribution", action="store_true")
    p.add_argument("--skip-ssim", action="store_true")
    p.add_argument("--no-softmax", dest="softmax", action="store_false")
    p.set_defaults(softmax=True)
    p.add_argument("--max-samples", type=int, default=None, help="Cap factual samples per dataset (debug)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASET_KEYS:
            raise SystemExit(f"Unknown dataset {d!r}")

    out = args.out.expanduser().resolve()
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    print(f"Harvest → {out}")
    t0 = time.time()

    if args.evolution_only:
        from harvest_evolution import harvest_evolution

        if args.dry_run:
            print(f"Would write evolution JSON under {out / 'evolution'} (no factual/metrics rewrite)")
            return 0
        evo = harvest_evolution(
            out,
            datasets,
            evolution_root=args.evolution_root,
            write_fixture_if_missing=args.write_fixture,
        )
        print(f"Evolution-only done in {time.time() - t0:.1f}s  datasets={list(evo)}")
        if not evo:
            print("No traces found. Point --evolution-root at evolution_experiment, or add --write-fixture.")
        else:
            idx_path = out / "index.json"
            if idx_path.is_file():
                try:
                    payload = json.loads(idx_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        payload["evolution"] = {k: True for k in evo}
                        _write_json(idx_path, payload)
                except (OSError, json.JSONDecodeError):
                    pass
            print(f"Wrote {out / 'evolution'}/. Other files under --out were not rewritten.")
            print("If --out is not already site/public/data, copy the evolution/ folder into that bundle.")
        if args.zip and (out / "evolution").is_dir():
            zip_bundle(out / "evolution", args.zip.expanduser().resolve(), arc_from=out)
        return 0

    names = {ds: class_names_for(ds) for ds in DATASET_KEYS}
    if not args.dry_run:
        _write_json(out / "paper.json", PAPER)
        _write_json(out / "catalog.json", catalog_payload())
        _write_json(out / "class_names.json", names)

    metrics_ok = export_metrics(out, datasets) if not args.dry_run else {ds: False for ds in datasets}

    missing: list[dict[str, Any]] = []
    index_datasets: dict[str, Any] = {}

    if args.write_fixture and not args.dry_run:
        write_fixture(out, args.webp_quality)

    evo_index: dict[str, Any] = {}
    if not args.skip_evolution and not args.dry_run:
        from harvest_evolution import harvest_evolution

        evo_index = harvest_evolution(
            out,
            datasets,
            evolution_root=args.evolution_root,
            write_fixture_if_missing=args.write_fixture,
        )

    if args.metrics_only:
        _write_json(
            out / "index.json",
            {
                "version": BUNDLE_VERSION,
                "generated_at": _now(),
                "datasets": datasets,
                "metrics": metrics_ok,
                "evolution": {k: True for k in evo_index},
                "mode": "metrics-only" + ("+fixture" if args.write_fixture else ""),
            },
        )
        _write_json(out / "missing.json", {"items": missing})
        print(f"Done in {time.time() - t0:.1f}s  size={_fmt_bytes(_dir_bytes(out))}")
        if args.zip:
            zip_bundle(out, args.zip.expanduser().resolve())
        return 0

    from afp.datasets import get_app_dataset_spec
    from afp.manifest import load_eval_manifest

    try:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = "cpu"
    print(f"  device={device}")

    for ds in datasets:
        spec = get_app_dataset_spec(ds)
        ds_info: dict[str, Any] = {
            "name": ds,
            "num_classes": spec.num_classes,
            "factual": 0,
            "blank": 0,
        }
        if not args.skip_factual:
            man_path = spec.default_manifest
            if not man_path.is_file():
                print(f"  [factual] {ds}: missing manifest {man_path}")
                missing.append({"dataset": ds, "kind": "manifest", "error": str(man_path)})
            else:
                entries = load_eval_manifest(man_path)
                if args.max_samples:
                    entries = entries[: args.max_samples]
                rows = []
                for i, ent in enumerate(entries):
                    print(f"  [factual] {ds} {i + 1}/{len(entries)} idx={ent.sample_idx}")
                    row = harvest_factual_sample(
                        spec=spec,
                        entry=ent,
                        names=names[ds],
                        out_dir=out / "factual" / ds,
                        args=args,
                        missing=missing,
                        device=device,
                    )
                    if row:
                        rows.append(row)
                if not args.dry_run:
                    _write_json(out / "factual" / ds / "index.json", {"dataset": ds, "samples": rows})
                ds_info["factual"] = len(rows)

        if not args.skip_blank:
            from afp.blank_paths import iter_target_classes

            targets = iter_target_classes(spec.default_blank_graco_root)
            if not targets:
                targets = list(range(spec.num_classes if ds != "imagenet100" else 0))
                if not targets:
                    print(f"  [blank] {ds}: no target_* folders")
            ssim_pool = None
            if targets and not args.skip_ssim and not args.dry_run and spec.default_data_root.is_dir():
                try:
                    from afp.blank_ssim_refs import cached_val_by_class

                    ssim_pool = cached_val_by_class(ds, str(spec.default_data_root), None)
                except Exception as exc:
                    print(f"  [blank] {ds}: SSIM pool skipped ({exc})")
            rows_b = []
            for k in targets:
                print(f"  [blank] {ds} target_{k:03d}")
                rows_b.append(
                    harvest_blank_target(
                        spec=spec,
                        k=k,
                        names=names[ds],
                        out_dir=out / "blank" / ds,
                        args=args,
                        missing=missing,
                        device=device,
                        ssim_pool=ssim_pool,
                    )
                )
            if not args.dry_run:
                _write_json(out / "blank" / ds / "index.json", {"dataset": ds, "targets": rows_b})
            ds_info["blank"] = len(rows_b)
        index_datasets[ds] = ds_info

    if args.dry_run:
        n_f = sum(v.get("factual", 0) for v in index_datasets.values())
        n_b = sum(v.get("blank", 0) for v in index_datasets.values())
        est = n_f * 700_000 + n_b * 400_000
        print(f"Dry-run factual={n_f} blank={n_b} estimated ~{_fmt_bytes(est)}")
        return 0

    _write_json(
        out / "index.json",
        {
            "version": BUNDLE_VERSION,
            "generated_at": _now(),
            "datasets": index_datasets,
            "metrics": metrics_ok,
            "evolution": {k: True for k in evo_index},
        },
    )
    _write_json(out / "missing.json", {"items": missing})
    size = _dir_bytes(out)
    print(f"Done in {time.time() - t0:.1f}s  size={_fmt_bytes(size)}  missing={len(missing)}")
    print("Extract this folder into site/public/data/ (or use --zip).")
    if args.zip:
        zip_bundle(out, args.zip.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
