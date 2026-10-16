#!/usr/bin/env python3
"""Batch explainability pipeline over eval manifest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from explainability.config import get_dataset_spec, resolve_graco_results
from explainability.data.manifest import list_manifest_entries
from explainability.io.artifacts import sample_artifacts_complete
from explainability.pipeline.sample import ALL_METHODS, process_sample
from explainability.plots.aggregate import (
    print_cross_dataset_srg_table,
    print_srg_mean_table,
    write_cross_dataset_plot,
    write_dataset_summary,
)
from explainability.srg.occlusion import OCCLUSION_STRATEGIES
from explainability.srg.prob_class import CLI_GRACO_SRG_PROB_MODES


def _say(msg: str) -> None:
    print(msg, flush=True)


def _skip_reason_line(reason: str | None) -> str:
    if not reason:
        return "unknown"
    return reason.splitlines()[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Graco explainability — batch")
    p.add_argument(
        "--dataset",
        action="append",
        choices=["mnist", "cifar10", "imagenet100"],
        default=None,
        help="Dataset to run (repeatable). Default: mnist only",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override eval_manifest.json path (applies to each --dataset run)",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--graco-results", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--no-overlays", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument(
        "--occlusion",
        choices=OCCLUSION_STRATEGIES,
        default=None,
        help="SRG pixel corruption strategy (default: config srg_occlusion)",
    )
    p.add_argument(
        "--occlusion-noise-std",
        type=float,
        default=None,
        help="Std dev for gaussian_noise occlusion in Alibi space (default: config)",
    )
    p.add_argument(
        "--srg-graco-prob-class",
        choices=CLI_GRACO_SRG_PROB_MODES,
        default=None,
        help="SRG class for Graco/phenotype methods: base=model pred on original, pred=explanation (default: config)",
    )
    p.add_argument("--no-srg-plots", action="store_true")
    p.add_argument(
        "--srg-infer-batch-size",
        type=int,
        default=None,
        help="SRG classifier batch size (default: config srg_infer_batch_size)",
    )
    p.add_argument(
        "--pagerank-backend",
        choices=["implicit", "scipy", "networkx"],
        default=None,
        help="PageRank backend for pagerank_cooccur (default: config pagerank_backend)",
    )
    p.add_argument(
        "--pagerank-damping",
        type=float,
        default=None,
        help="PageRank damping factor (default: config pagerank_damping)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Reprocess samples even when output artifacts already exist",
    )
    args = p.parse_args()

    datasets = args.dataset or ["mnist"]
    device = torch.device(args.device)
    methods = args.methods or ALL_METHODS
    save_overlays = not args.no_overlays
    save_srg_plots = not args.no_srg_plots
    processed = 0
    skipped = 0
    already_complete = 0

    _say(f"Batch explainability: datasets={datasets} limit={args.limit} device={args.device}")
    _say("Note: batch iterates eval_manifest.json (unlike run_sample which takes --sample-idx/--y-true).")

    for ds_name in datasets:
        spec = get_dataset_spec(ds_name)
        manifest_path = args.manifest or spec.manifest
        _say("")
        _say(f"=== {ds_name} ===")
        _say(f"  manifest: {manifest_path.resolve()}")

        try:
            entries = list_manifest_entries(spec, limit=args.limit, manifest_path=manifest_path)
        except FileNotFoundError as exc:
            _say(f"  ERROR: {exc}")
            continue
        except (KeyError, ValueError) as exc:
            _say(f"  ERROR: bad manifest: {exc}")
            continue

        if not entries:
            _say("  WARNING: manifest has 0 entries — nothing to process for this dataset.")
            continue

        _say(f"  entries: {len(entries)}")

        if args.graco_results is not None:
            graco_root = resolve_graco_results(spec, args.graco_results)
        else:
            first = entries[0]
            graco_root = resolve_graco_results(
                spec,
                None,
                sample_idx=first.sample_idx,
                y_true=first.y_true,
            )
        weights = args.weights or spec.weights
        _say(f"  graco_root: {graco_root.resolve()}")
        _say(f"  weights: {weights.resolve()}")

        ds_processed = 0
        ds_skipped = 0
        ds_already_complete = 0
        n_entries = len(entries)

        for i, entry in enumerate(entries, start=1):
            if not args.force:
                complete, missing = sample_artifacts_complete(
                    ds_name,
                    entry.sample_idx,
                    methods,
                    save_overlays=save_overlays,
                    save_srg_plots=save_srg_plots,
                )
                if complete:
                    already_complete += 1
                    ds_already_complete += 1
                    _say(
                        f"  [{i}/{n_entries}] DONE sample={entry.sample_idx} "
                        f"y_true={entry.y_true} (artifacts already present)"
                    )
                    continue

            result = process_sample(
                spec,
                entry.sample_idx,
                entry.y_true,
                graco_root,
                weights,
                device,
                methods=methods,
                save_overlays=save_overlays,
                occlusion_strategy=args.occlusion,
                occlusion_noise_std=args.occlusion_noise_std,
                srg_graco_prob_class=args.srg_graco_prob_class,
                infer_batch_size=args.srg_infer_batch_size,
                pagerank_backend=args.pagerank_backend,
                pagerank_damping=args.pagerank_damping,
                save_srg_plots=not args.no_srg_plots,
            )
            if result.skipped:
                skipped += 1
                ds_skipped += 1
                _say(
                    f"  [{i}/{n_entries}] SKIP sample={entry.sample_idx} y_true={entry.y_true}: "
                    f"{_skip_reason_line(result.skip_reason)}"
                )
                logging.info(
                    "Skip sample %s (%s): %s",
                    entry.sample_idx,
                    ds_name,
                    result.skip_reason,
                )
            else:
                processed += 1
                ds_processed += 1
                _say(
                    f"  [{i}/{n_entries}] OK sample={entry.sample_idx} "
                    f"class={result.explanation_class} n_pheno={result.n_phenotypes} "
                    f"srg_methods={len(result.srg_scores)}"
                )
                logging.info(
                    "Done sample %s (%s) SRG methods=%d",
                    entry.sample_idx,
                    ds_name,
                    len(result.srg_scores),
                )

        _say(
            f"  dataset done: processed={ds_processed} skipped={ds_skipped} "
            f"already_complete={ds_already_complete}"
        )
        print_srg_mean_table(ds_name)

        if not args.skip_plots:
            write_dataset_summary(ds_name)

    if len(datasets) > 1:
        print_cross_dataset_srg_table(datasets)

    if not args.skip_plots and len(datasets) > 1:
        write_cross_dataset_plot(datasets)

    _say("")
    _say(
        f"Finished: processed={processed} skipped={skipped} "
        f"already_complete={already_complete}"
    )
    if processed == 0 and skipped == 0 and already_complete == 0:
        _say(
            "No samples were processed. Check that eval_manifest.json exists and is non-empty, "
            "and pass --dataset mnist --limit 5 for a quick test."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
