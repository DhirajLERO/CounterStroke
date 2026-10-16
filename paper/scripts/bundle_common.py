#!/usr/bin/env python3
"""Shared paths, naming and helpers for the results-bundle pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "results_bundle"
RAW = BUNDLE / "raw"
CONSOLIDATED = BUNDLE / "consolidated"
STATS = BUNDLE / "stats"
MANIFEST_PATH = BUNDLE / "BUNDLE_MANIFEST.json"

BUNDLE_VERSION = 1


DATASETS: list[str] = ["MNIST", "CIFAR-10", "ImageNet-100"]

# directory slug used by the explainability outputs -> dataset label
SRG_DATASET_LABELS: dict[str, str] = {
    "mnist": "MNIST",
    "cifar10": "CIFAR-10",
    "imagenet100": "ImageNet-100",
}

METHOD_LABELS: dict[str, str] = {
    "graco_base_o2": "CS_sparse",
    "graco_base_o3": "CS_prox",
    "graco50": "CS_50",
    "counterfactual_pt": "GradCF",
    "cfproto": "CFProto",
    "cfrl": "CFRL",
    "leapfactual": "LeapFactual",
    "leapfactual_r": "LeapFactual-R",
}

METHOD_ORDER: list[str] = [
    "CS_sparse",
    "CS_prox",
    "CS_50",
    "GradCF",
    "CFProto",
    "CFRL",
    "LeapFactual",
    "LeapFactual-R",
]

CS_METHODS: set[str] = {"CS_sparse", "CS_prox", "CS_50"}

METHOD_FAMILY: dict[str, str] = {
    "CS_sparse": "counterstroke",
    "CS_prox": "counterstroke",
    "CS_50": "counterstroke",
    "GradCF": "pixel-gradient",
    "CFProto": "prototype",
    "CFRL": "generative-rl",
    "LeapFactual": "generative-flow",
    "LeapFactual-R": "generative-flow",
}

METHOD_IS_TARGETED: dict[str, bool] = {
    "CS_sparse": False,
    "CS_prox": False,
    "CS_50": False,
    "GradCF": False,
    "CFProto": True,
    "CFRL": True,
    "LeapFactual": True,
    "LeapFactual-R": True,
}

SRG_METHOD_LABELS: dict[str, str] = {
    "spatial_coherence": "Spatial coherence",
    "change_frequency": "Change frequency",
    "cochange_strict": "Co-change",
    "sum_intensity": "Sum intensity",
    "mean_intensity": "Mean intensity",
    "pagerank_cooccur": "PageRank co-occurrence",
    "integrated_gradients": "Integrated Gradients",
    "input_x_gradient": "Input x Gradient",
    "smoothgrad": "SmoothGrad",
    "occlusion": "Occlusion",
    "gradcam": "Grad-CAM",
    "shap_gradient": "SHAP (gradient)",
}

SRG_PHENOTYPE_METHODS: set[str] = {
    "Spatial coherence",
    "Change frequency",
    "Co-change",
    "Sum intensity",
    "Mean intensity",
    "PageRank co-occurrence",
}

SRG_METHOD_ORDER: list[str] = [
    "Spatial coherence",
    "Change frequency",
    "Co-change",
    "Sum intensity",
    "Mean intensity",
    "PageRank co-occurrence",
    "Integrated Gradients",
    "Input x Gradient",
    "SmoothGrad",
    "Occlusion",
    "Grad-CAM",
    "SHAP (gradient)",
]

# per-sample metrics carried through consolidation; True == higher is better
METRIC_HIGHER_BETTER: dict[str, bool] = {
    "l2_distance": False,
    "sparsity_count": False,
    "sparsity_frac": False,
    "ssim_to_original": True,
    "lpips_to_original": False,
    "ssim_to_random_pred_class": True,
    "lpips_to_random_pred_class": False,
    "p_pred_cf": True,
    "p_true_cf": False,
    "delta_p_true": False,
    "p_intended_target_cf": True,
    "inception_sqdist_to_source_class_mean": False,
    "inception_sqdist_to_predicted_class_mean": False,
    "embedding_margin_src_minus_pred": True,
}

# metrics grouped into multiple-comparison families (one family per dataset)
METRIC_FAMILIES: dict[str, list[str]] = {
    "proximity_perceptual": [
        "l2_distance",
        "sparsity_count",
        "sparsity_frac",
        "ssim_to_original",
        "lpips_to_original",
    ],
    "class_reference": [
        "ssim_to_random_pred_class",
        "lpips_to_random_pred_class",
    ],
    "confidence": ["p_pred_cf"],
    "embedding": [
        "inception_sqdist_to_source_class_mean",
        "inception_sqdist_to_predicted_class_mean",
        "embedding_margin_src_minus_pred",
    ],
}

# subsets used for aggregation; documented in the consolidated outputs
SUBSETS: list[str] = ["own_flips", "common_success", "available", "attempted"]

KNOWN_GAPS: list[str] = [
    "Evolutionary run artifacts (final populations, per-generation logs) are "
    "excluded on purpose: multiple runs are pending.",
    "Wall-clock runtime and classifier-query counts per image are not recorded "
    "in any artifact; the cost analysis must come from the pending runs.",
    "Ablation results (random strokes, single-objective search, grammar-free "
    "black-box search) do not exist yet.",
    "Per-sample counterfactual tensors (tensors.npz) are absent, so qualitative "
    "factual/counterfactual grids cannot be rendered from this bundle. Re-run "
    "gather with --include-tensors where they exist.",
    "Blank-canvas evolution outputs (analysis/*/outputs) are absent, so the "
    "blank-canvas section has no quantitative summary in this bundle.",
    "Phenotype/stroke statistics (stroke count, genotype length) are not "
    "recorded outside the evolution artifacts.",
]

ALPHA = 0.05
Z_975 = 1.959963984540054



def setup_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("bundle")


LOG = logging.getLogger("bundle")



def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def csv_row_count(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOG.info("wrote %s (%d rows, %d cols)", rel(path), len(df), df.shape[1])
    return path


def write_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    LOG.info("wrote %s (%d chars)", rel(path), len(text))
    return path


def write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    LOG.info("wrote %s", rel(path))
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "bundle_version": BUNDLE_VERSION,
        "created_utc": utc_now(),
        "source_root": str(ROOT),
        "git_commit": git_commit(),
        "stages": {},
        "files": [],
        "missing_sources": [],
        "known_gaps": KNOWN_GAPS,
    }


def save_manifest(manifest: dict[str, Any]) -> Path:
    return write_json(manifest, MANIFEST_PATH)


def manifest_entry(path: Path, source: Path | None = None, stage: str = "") -> dict[str, Any]:
    return {
        "path": str(path.relative_to(BUNDLE)),
        "source": rel(source) if source is not None else None,
        "stage": stage,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "rows": csv_row_count(path),
    }


def record_stage(
    stage: str,
    outputs: Sequence[Path],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register stage outputs in the bundle manifest, replacing prior entries."""
    manifest = read_manifest()
    entries = [manifest_entry(p, stage=stage) for p in outputs if p.exists()]
    manifest["files"] = [f for f in manifest.get("files", []) if f.get("stage") != stage]
    manifest["files"].extend(entries)
    manifest["stages"][stage] = {
        "completed_utc": utc_now(),
        "n_files": len(entries),
        "bytes": int(sum(e["bytes"] for e in entries)),
        **(extra or {}),
    }
    manifest["known_gaps"] = KNOWN_GAPS
    save_manifest(manifest)
    return manifest




def wilson_interval(k: int, n: int, z: float = Z_975) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def holm_correct(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaNs preserved)."""
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    adjusted = np.full(p.shape, np.nan)
    idx = np.flatnonzero(finite)
    if idx.size == 0:
        return adjusted.tolist()
    order = idx[np.argsort(p[idx], kind="stable")]
    m = idx.size
    running = 0.0
    for rank, i in enumerate(order):
        value = (m - rank) * p[i]
        running = max(running, value)
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def canonical_methods(df: pd.DataFrame, column: str = "method") -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column].map(lambda m: METHOD_LABELS.get(m, m))
    return out


def order_methods(values: Iterable[str]) -> list[str]:
    seen = list(dict.fromkeys(values))
    known = [m for m in METHOD_ORDER if m in seen]
    return known + [m for m in seen if m not in METHOD_ORDER]


def order_datasets(values: Iterable[str]) -> list[str]:
    seen = list(dict.fromkeys(values))
    known = [d for d in DATASETS if d in seen]
    return known + [d for d in seen if d not in DATASETS]
