"""Evolve Graco counterfactuals on MorphoMNIST eval subset (three variants per slot)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_MORPHO = Path(__file__).resolve().parent
_BENCHMARK = _MORPHO.parent
_SHARED = _BENCHMARK / "shared"
_GRACOIMAGE = _BENCHMARK.parent
sys.path.insert(0, str(_SHARED))
sys.path.insert(0, str(_GRACOIMAGE))

from paths import setup_vendor_path  # noqa: E402
from eval_subset import load_eval_indices  # noqa: E402
from eval_interventions import load_intervention_manifest, manifest_intervention_tensors, lookup_manifest_entry  # noqa: E402
from eval_protocol import DEFAULT_EVAL_SAMPLES, GRACO_VARIANTS, interventions_path_for_indices  # noqa: E402
from image_io import morphomnist_to_turtle, turtle_to_morphomnist  # noqa: E402
from graco_predictors import load_predictors  # noqa: E402
from graco_variant_pick import pick_benchmark_variants  # noqa: E402
import graco_turtle_core as gtc  # noqa: E402

setup_vendor_path()
from datasets.transforms import ReturnDictTransform  # noqa: E402
from datasets.morphomnist.dataset import MorphoMNISTLike, unnormalize as unnormalize_morphomnist  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scm-config", type=Path, default=_MORPHO / "configs" / "vae.json")
    ap.add_argument("--classifier-config", type=Path, default=_MORPHO / "configs" / "classifier.json")
    ap.add_argument("--eval-indices", type=Path, default=_MORPHO / "data_subset" / "eval_indices.json")
    ap.add_argument("--interventions", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=_MORPHO / "outputs" / "graco")
    ap.add_argument("--max-samples", type=int, default=DEFAULT_EVAL_SAMPLES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.scm_config) as f:
        scm_cfg = json.load(f)
    attribute_size = scm_cfg["attribute_size"]
    transform = ReturnDictTransform(attribute_size)
    test_set = MorphoMNISTLike(attribute_size, split="test", transform=transform)

    interventions_path = args.interventions or interventions_path_for_indices(args.eval_indices)
    if not interventions_path.is_file():
        raise SystemExit(f"Missing intervention manifest: {interventions_path}. Run build_eval_subset.py first.")
    manifest = load_intervention_manifest(interventions_path)
    indices = load_eval_indices(args.eval_indices)[: args.max_samples]

    predictors, _ = load_predictors(args.classifier_config, scm_cfg, device=args.device)
    gtc.x_max = gtc.y_max = 32
    gtc.clip_lo, gtc.clip_hi = -1.0, 1.0
    gtc.n_channels = 1
    grammar = _MORPHO / "graco" / "grammar" / "AG_turtle_morphomnist.bnf"
    graco_cfg = json.loads((_MORPHO / "graco_config.json").read_text()) if (_MORPHO / "graco_config.json").is_file() else {}

    for variant in GRACO_VARIANTS:
        (args.out_dir / variant).mkdir(parents=True, exist_ok=True)

    for idx in indices:
        batch = test_set[idx]
        if not isinstance(batch, dict):
            img, attrs = batch
            batch = transform(img, attrs)
        factual = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        img_np = morphomnist_to_turtle(factual["image"].squeeze(0))

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
                dataset_name="morphomnist",
                grammar_path=grammar,
                config=graco_cfg,
                device=args.device,
            )

            def image_from_phenotype(arr_np):
                t = turtle_to_morphomnist(arr_np, factual["image"].squeeze(0))
                return t.unsqueeze(0)

            picks = pick_benchmark_variants(
                candidates,
                do_parent=do_parent,
                targets=targets,
                predictors=predictors,
                dataset_name="morphomnist",
                unnormalize_fn=unnormalize_morphomnist,
                image_from_phenotype=image_from_phenotype,
            )

            for variant, cand in picks.items():
                if cand is None:
                    continue
                cf_tensor = turtle_to_morphomnist(cand.image_np, factual["image"].squeeze(0))
                out = {**targets, "image": cf_tensor.unsqueeze(0)}
                meta = {
                    "sample_idx": idx,
                    "do_parent": do_parent,
                    "variant": variant,
                    "o1": cand.o1,
                    "o2": cand.o2,
                    "o3": cand.o3,
                    "phenotype": cand.phenotype,
                }
                path = args.out_dir / variant / f"cf_{idx:05d}_{do_parent}.pt"
                torch.save({**out, "meta": meta}, path)
                print(f"saved {variant} cf_{idx:05d}_{do_parent}.pt")


if __name__ == "__main__":
    main()
