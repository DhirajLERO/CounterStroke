#!/usr/bin/env python3
"""Run explainability pipeline for a single sample."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from explainability.config import candidate_graco_roots, get_dataset_spec, resolve_graco_results
from explainability.pipeline.sample import ALL_METHODS, process_sample
from explainability.srg.occlusion import OCCLUSION_STRATEGIES
from explainability.srg.prob_class import CLI_GRACO_SRG_PROB_MODES


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Graco explainability — single sample")
    p.add_argument("--dataset", required=True, choices=["mnist", "cifar10", "imagenet100"])
    p.add_argument("--sample-idx", type=int, required=True)
    p.add_argument("--y-true", type=int, required=True)
    p.add_argument("--target-class", type=int, default=None)
    p.add_argument("--graco-results", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--methods", nargs="*", default=None, help=f"Subset of: {ALL_METHODS}")
    p.add_argument("--no-overlays", action="store_true")
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
        "--print-paths",
        action="store_true",
        help="Print candidate Graco results roots and exit",
    )
    args = p.parse_args()

    spec = get_dataset_spec(args.dataset)

    if args.print_paths:
        print(f"Dataset: {spec.name}")
        print("Candidate Graco results roots (first existing is used by default):")
        for root in candidate_graco_roots(spec):
            status = "exists" if root.is_dir() else "missing"
            print(f"  [{status}] {root.resolve()}")
        sys.exit(0)

    graco_root = resolve_graco_results(
        spec,
        args.graco_results,
        sample_idx=args.sample_idx,
        y_true=args.y_true,
    )
    weights = args.weights or spec.weights
    device = torch.device(args.device)

    logging.info("Using graco results root: %s", graco_root)

    result = process_sample(
        spec,
        args.sample_idx,
        args.y_true,
        graco_root,
        weights,
        device,
        target_class=args.target_class,
        methods=args.methods,
        save_overlays=not args.no_overlays,
        occlusion_strategy=args.occlusion,
        occlusion_noise_std=args.occlusion_noise_std,
        srg_graco_prob_class=args.srg_graco_prob_class,
        infer_batch_size=args.srg_infer_batch_size,
        pagerank_backend=args.pagerank_backend,
        pagerank_damping=args.pagerank_damping,
        save_srg_plots=not args.no_srg_plots,
    )
    if result.skipped:
        print(f"SKIPPED: {result.skip_reason}")
        sys.exit(1)
    print(f"OK sample={result.sample_idx} class={result.explanation_class} n_pheno={result.n_phenotypes}")
    for m, s in sorted(result.srg_scores.items()):
        print(f"  {m}: SRG={s:.4f}")


if __name__ == "__main__":
    main()
