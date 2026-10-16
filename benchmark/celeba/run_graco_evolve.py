"""Evolve Graco counterfactuals on CelebA eval subset (three variants per slot)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_CELEBA = Path(__file__).resolve().parent
_BENCHMARK = _CELEBA.parent
_SHARED = _BENCHMARK / "shared"
sys.path.insert(0, str(_SHARED))

from paths import setup_vendor_path  # noqa: E402
from eval_subset import load_eval_indices  # noqa: E402
from eval_interventions import load_intervention_manifest, manifest_intervention_tensors, lookup_manifest_entry  # noqa: E402
from eval_protocol import DEFAULT_EVAL_SAMPLES, GRACO_VARIANTS, interventions_path_for_indices  # noqa: E402
from image_io import celeba_to_turtle, turtle_to_celeba  # noqa: E402
from graco_predictors import load_predictors  # noqa: E402
from graco_variant_pick import pick_benchmark_variants  # noqa: E402
import graco_turtle_core as gtc  # noqa: E402

setup_vendor_path()
from datasets.transforms import ReturnDictTransform  # noqa: E402
from datasets.celeba.dataset import Celeba, unnormalize as unnormalize_celeba  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=["simple", "complex"], default="simple")
    ap.add_argument("--scm-config", type=Path, default=None)
    ap.add_argument("--classifier-config", type=Path, default=None)
    ap.add_argument("--eval-indices", type=Path, default=None)
    ap.add_argument("--interventions", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    graph = args.graph
    cfg_dir = _CELEBA / "configs" / graph
    scm_cfg_path = args.scm_config or cfg_dir / "vae.json"
    clf_cfg_path = args.classifier_config or cfg_dir / "classifier.json"
    eval_indices = args.eval_indices or _CELEBA / "data_subset" / f"eval_indices_{graph}.json"
    out_dir = args.out_dir or _CELEBA / "outputs" / "graco" / graph

    with open(scm_cfg_path) as f:
        scm_cfg = json.load(f)
    attribute_size = scm_cfg["attribute_size"]
    transform = ReturnDictTransform(attribute_size)
    data_dir = str(_CELEBA / "data")
    test_set = Celeba(attribute_size, split="test", transform=transform, data_dir=data_dir)

    interventions_path = args.interventions or interventions_path_for_indices(eval_indices)
    if not interventions_path.is_file():
        raise SystemExit(f"Missing intervention manifest: {interventions_path}. Run build_eval_subset.py first.")
    manifest = load_intervention_manifest(interventions_path)
    indices = load_eval_indices(eval_indices)[: args.max_samples]

    predictors, _ = load_predictors(clf_cfg_path, scm_cfg, device=args.device)
    gtc.x_max = gtc.y_max = 64
    gtc.clip_lo, gtc.clip_hi = 0.0, 1.0
    grammar = _CELEBA / "graco" / "grammar" / "AG_turtle_celeba.bnf"
    graco_cfg_path = _CELEBA / f"graco_config_{graph}.json"
    graco_cfg = json.loads(graco_cfg_path.read_text()) if graco_cfg_path.is_file() else {}

    for variant in GRACO_VARIANTS:
        (out_dir / variant).mkdir(parents=True, exist_ok=True)

    for idx in indices:
        batch = test_set[idx]
        if not isinstance(batch, dict):
            img, attrs = batch
            batch = transform(img, attrs)
        factual = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim <= 1 else v) for k, v in batch.items()}
        if factual["image"].ndim == 3:
            factual["image"] = factual["image"].unsqueeze(0)
        img_np = celeba_to_turtle(factual["image"].squeeze(0))

        for do_parent in attribute_size:
            entry = lookup_manifest_entry(manifest, idx, do_parent)
            if entry is None:
                continue
            inter = manifest_intervention_tensors(entry, device=args.device)
            targets = {**factual, **inter}

            candidates = gtc.evolve_benchmark_image(
                image_np=img_np,
                targets=targets,
                predictors=predictors,
                dataset_name="celeba",
                grammar_path=grammar,
                config=graco_cfg,
                device=args.device,
            )

            def image_from_phenotype(arr_np):
                t = turtle_to_celeba(arr_np, factual["image"].squeeze(0))
                return t.unsqueeze(0)

            picks = pick_benchmark_variants(
                candidates,
                do_parent=do_parent,
                targets=targets,
                predictors=predictors,
                dataset_name="celeba",
                unnormalize_fn=unnormalize_celeba,
                image_from_phenotype=image_from_phenotype,
            )

            for variant, cand in picks.items():
                if cand is None:
                    continue
                cf_tensor = turtle_to_celeba(cand.image_np, factual["image"].squeeze(0))
                out = {**targets, "image": cf_tensor.unsqueeze(0)}
                meta = {
                    "sample_idx": idx,
                    "do_parent": do_parent,
                    "variant": variant,
                    "graph": graph,
                    "o1": cand.o1,
                    "o2": cand.o2,
                    "o3": cand.o3,
                    "phenotype": cand.phenotype,
                }
                path = out_dir / variant / f"cf_{idx:05d}_{do_parent}.pt"
                torch.save({**out, "meta": meta}, path)
                print(f"saved {graph} {variant} cf_{idx:05d}_{do_parent}.pt")


if __name__ == "__main__":
    main()
