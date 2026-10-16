"""Process one manifest sample end-to-end."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..baselines.classifier import predict_argmax_class
from ..baselines.methods import BASELINE_METHODS, compute_baseline
from ..config import DatasetSpec, get_defaults
from ..data.original_class import normalize_record_original_classes, resolve_original_class
from ..data.population import load_final_population
from ..data.selection import select_explanation_phenotypes
from ..data.graco_paths import find_graco_sample_dir, graco_sample_dir_hint, run_dir
from ..graph.pagerank import pagerank_cooccur_map
from ..io.artifacts import (
    load_diffs_cache,
    sample_output_dir,
    save_diffs_cache,
    save_map,
    save_meta,
    save_overlay,
    save_srg,
)
from ..phenotype.aggregations import PHENOTYPE_METHODS, heat_from_diffs
from ..phenotype.diffs import collect_diffs, importance_to_ranking
from ..plots.srg_curves import save_mif_lif_grid, save_mif_lif_plot
from ..render.turtle import load_original_alibi
from ..srg.evaluate import SRGResult, compute_srg
from ..srg.prob_class import srg_prob_class_for_method

logger = logging.getLogger(__name__)

ALL_PHENOTYPE_METHODS = PHENOTYPE_METHODS + ["pagerank_cooccur"]
GRACO_SRG_METHODS = set(ALL_PHENOTYPE_METHODS)
ALL_METHODS = ALL_PHENOTYPE_METHODS + list(BASELINE_METHODS.keys())


@dataclass
class SampleResult:
    sample_idx: int
    skipped: bool
    skip_reason: str | None
    explanation_class: int | None
    n_phenotypes: int
    srg_scores: dict[str, float]


def _display_hw(spec: DatasetSpec, img_alibi: np.ndarray) -> np.ndarray:
    arr = np.asarray(img_alibi, dtype=np.float32)
    if spec.name == "imagenet100" and arr.ndim == 3 and arr.shape[0] == 3:
        hwc = np.transpose(arr + 0.5, (1, 2, 0))
        return np.clip(hwc, 0, 1)
    if arr.ndim == 2:
        return np.clip(arr + 0.5, 0, 1)
    return np.clip(arr + 0.5, 0, 1)


def process_sample(
    spec: DatasetSpec,
    sample_idx: int,
    y_true: int,
    graco_results_root: Path,
    weights: Path,
    device: torch.device,
    *,
    target_class: int | None = None,
    methods: list[str] | None = None,
    prob_threshold: float | None = None,
    max_render: int | None = None,
    curve_points: int | None = None,
    occlusion_strategy: str | None = None,
    occlusion_noise_std: float | None = None,
    srg_graco_prob_class: str | None = None,
    infer_batch_size: int | None = None,
    pagerank_backend: str | None = None,
    pagerank_damping: float | None = None,
    save_overlays: bool = True,
    save_srg_plots: bool = True,
    run_id: int = 0,
) -> SampleResult:
    defaults = get_defaults()
    prob_threshold = prob_threshold if prob_threshold is not None else float(defaults.get("prob_threshold", 0.5))
    max_render = max_render if max_render is not None else int(defaults.get("max_render", 200))
    curve_points = curve_points if curve_points is not None else int(defaults.get("srg_curve_points", 100))
    infer_batch_size = (
        infer_batch_size
        if infer_batch_size is not None
        else int(defaults.get("srg_infer_batch_size", 32))
    )
    pagerank_backend = pagerank_backend or str(defaults.get("pagerank_backend", "scipy"))
    pagerank_damping = (
        pagerank_damping
        if pagerank_damping is not None
        else float(defaults.get("pagerank_damping", 0.85))
    )
    occlusion_strategy = occlusion_strategy or str(defaults.get("srg_occlusion", "mean_alibi"))
    occlusion_noise_std = (
        occlusion_noise_std
        if occlusion_noise_std is not None
        else float(defaults.get("srg_occlusion_noise_std", 0.15))
    )
    srg_graco_prob_class = srg_graco_prob_class or str(defaults.get("srg_graco_prob_class", "base"))
    cochange_tau = float(defaults.get("cochange_tau", 0.5))
    spatial_sigma = float(defaults.get("spatial_sigma", 1.5))
    eps = float(defaults.get("eps", 1e-6))

    methods = methods or ALL_METHODS
    out_dir = sample_output_dir(spec.name, sample_idx)

    sample_dir = find_graco_sample_dir(graco_results_root, sample_idx, y_true)
    if sample_dir is None:
        hint = graco_sample_dir_hint(graco_results_root, sample_idx, y_true)
        return SampleResult(sample_idx, True, f"graco_sample_dir_not_found\n{hint}", None, 0, {})

    rd = run_dir(sample_dir, run_id)
    records, source = load_final_population(sample_dir, run_id=run_id)
    if not records:
        return SampleResult(sample_idx, True, f"empty_population ({source})", None, 0, {})

    orig = load_original_alibi(rd, spec, run_id=run_id)
    if orig is None:
        return SampleResult(sample_idx, True, "original_image_missing", None, 0, {})

    original_class = resolve_original_class(rd, records, y_true, run_id=run_id)
    pickle_missing_label = source == "pkl" and all(r.original_class < 0 for r in records)
    normalize_record_original_classes(records, original_class)
    if pickle_missing_label:
        logger.info(
            "Resolved original_class=%s from summary/manifest (pickle omitted label)",
            original_class,
        )

    base_class = predict_argmax_class(spec, orig, weights, device)
    logger.info(
        "SRG base class (model pred on original): %s (y_true=%s)",
        base_class,
        original_class,
    )

    selection = select_explanation_phenotypes(
        records,
        prob_threshold=prob_threshold,
        target_class=target_class,
        original_class=original_class,
    )
    if not selection.records or selection.explanation_class is None:
        return SampleResult(sample_idx, True, "no_explanation_phenotypes", None, 0, {})

    explanation_class = selection.explanation_class
    diffs_cache = out_dir / "diffs.npz"
    if diffs_cache.is_file():
        try:
            diffs = load_diffs_cache(diffs_cache)
            shape = (
                diffs[0].shape
                if diffs
                else (orig.shape[1], orig.shape[2]) if orig.ndim == 3 and orig.shape[0] == 3 else orig.shape
            )
            n_used = len(diffs)
        except (KeyError, ValueError, OSError) as exc:
            logger.warning("Ignoring corrupt diffs cache %s: %s", diffs_cache, exc)
            diffs_cache.unlink(missing_ok=True)
            diffs, n_used, shape = collect_diffs(
                spec,
                orig,
                selection.records,
                target_pred_class=explanation_class,
                max_render=max_render,
                eps=eps,
            )
            if diffs:
                save_diffs_cache(diffs_cache, diffs)
    else:
        diffs, n_used, shape = collect_diffs(
            spec,
            orig,
            selection.records,
            target_pred_class=explanation_class,
            max_render=max_render,
            eps=eps,
        )
        if diffs:
            save_diffs_cache(diffs_cache, diffs)

    if not diffs:
        return SampleResult(sample_idx, True, "no_valid_diffs", explanation_class, 0, {})

    importance_maps: dict[str, np.ndarray] = {}

    for m in PHENOTYPE_METHODS:
        if m not in methods:
            continue
        importance_maps[m] = heat_from_diffs(
            diffs,
            shape,
            m,
            cochange_tau=cochange_tau,
            spatial_sigma=spatial_sigma,
            eps=eps,
        )

    if "pagerank_cooccur" in methods:
        importance_maps["pagerank_cooccur"] = pagerank_cooccur_map(
            diffs,
            shape,
            eps=eps,
            damping=pagerank_damping,
            backend=pagerank_backend,
        )

    for m in BASELINE_METHODS:
        if m not in methods:
            continue
        try:
            importance_maps[m] = compute_baseline(
                m, spec, orig, explanation_class, weights, device
            )
        except Exception as exc:
            logger.warning("Baseline %s failed for sample %s: %s", m, sample_idx, exc)

    srg_scores: dict[str, float] = {}
    srg_results: dict[str, SRGResult] = {}
    srg_prob_classes: dict[str, int] = {}
    base_hw = _display_hw(spec, orig)

    for method, imp in importance_maps.items():
        ranking = importance_to_ranking(imp)
        save_map(out_dir / "maps" / f"{method}.npz", imp, ranking)
        prob_class = srg_prob_class_for_method(
            method,
            base_class=base_class,
            explanation_class=explanation_class,
            graco_methods=GRACO_SRG_METHODS,
            graco_mode=srg_graco_prob_class,
        )
        srg_prob_classes[method] = prob_class
        try:
            srg = compute_srg(
                spec,
                orig,
                imp,
                prob_class,
                weights,
                device,
                curve_points=curve_points,
                occlusion_strategy=occlusion_strategy,
                occlusion_noise_std=occlusion_noise_std,
                rng_seed=sample_idx,
                infer_batch_size=infer_batch_size,
            )
            save_srg(out_dir / "srg" / f"{method}.npz", srg)
            srg_scores[method] = srg.srg_score
            srg_results[method] = srg
            if save_srg_plots:
                try:
                    save_mif_lif_plot(
                        out_dir / "srg_plots" / f"{method}.png",
                        srg,
                        method=method,
                        prob_class=prob_class,
                    )
                except Exception as exc:
                    logger.warning("SRG plot failed for %s sample %s: %s", method, sample_idx, exc)
        except Exception as exc:
            logger.warning("SRG failed for %s sample %s: %s", method, sample_idx, exc)
        if save_overlays:
            try:
                save_overlay(out_dir / "overlays" / f"{method}.png", base_hw, imp)
            except Exception:
                pass

    if save_srg_plots and srg_results:
        try:
            save_mif_lif_grid(
                out_dir / "srg_plots" / "_all_methods.png",
                srg_results,
                prob_classes=srg_prob_classes,
            )
        except Exception as exc:
            logger.warning("SRG grid plot failed for sample %s: %s", sample_idx, exc)

    save_meta(
        out_dir / "meta.json",
        {
            "dataset": spec.name,
            "sample_idx": sample_idx,
            "y_true": y_true,
            "original_class": original_class,
            "base_class": base_class,
            "explanation_class": explanation_class,
            "srg_graco_prob_class": srg_graco_prob_class,
            "srg_prob_classes": srg_prob_classes,
            "srg_occlusion": occlusion_strategy,
            "srg_occlusion_noise_std": occlusion_noise_std,
            "n_total": selection.n_total,
            "n_filtered": selection.n_filtered,
            "n_phenotypes": len(selection.records),
            "n_diffs_used": n_used,
            "class_counts": selection.class_counts,
            "population_source": source,
            "srg_scores": srg_scores,
        },
    )

    return SampleResult(
        sample_idx=sample_idx,
        skipped=False,
        skip_reason=None,
        explanation_class=explanation_class,
        n_phenotypes=len(selection.records),
        srg_scores=srg_scores,
    )
