"""Run benchmark metrics on saved Graco counterfactual bundles (aligned protocol)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_SHARED = Path(__file__).resolve().parent
_BENCHMARK = _SHARED.parent
sys.path.insert(0, str(_SHARED))
sys.path.insert(0, str(_BENCHMARK.parent))

from paths import setup_vendor_path, dataset_root  # noqa: E402

setup_vendor_path()

from datasets.transforms import ReturnDictTransform  # noqa: E402
from datasets.morphomnist.dataset import MorphoMNISTLike, unnormalize as unnormalize_morphomnist  # noqa: E402
from datasets.celeba.dataset import Celeba, unnormalize as unnormalize_celeba  # noqa: E402
from evaluation.metrics.effectiveness import effectiveness  # noqa: E402
from evaluation.metrics.fid import fid  # noqa: E402
from evaluation.metrics.minimality import minimality  # noqa: E402
from evaluation.embeddings.embeddings import get_embedding_model, get_embedding_fn  # noqa: E402
from graco_predictors import load_predictors  # noqa: E402
from graco_variant_pick import _batch_on_predictor_device  # noqa: E402
from uncond_vae_paths import apply_uncond_vae_env  # noqa: E402


def load_cf_bundles(path: Path) -> list[dict]:
    rows = []
    for p in sorted(path.glob("cf_*.pt")):
        rows.append(torch.load(p, map_location="cpu", weights_only=False))
    return rows


def _batch_from_bundle(b: dict) -> dict:
    cf_batch = {
        k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim == 1 else v
        for k, v in b.items()
        if k != "meta"
    }
    if "image" in cf_batch and cf_batch["image"].ndim == 3:
        cf_batch["image"] = cf_batch["image"].unsqueeze(0)
    return cf_batch


def _factual_batch(test_set, idx: int, transform) -> dict:
    batch = test_set[idx]
    if not isinstance(batch, dict):
        img, attrs = batch
        batch = transform(img, attrs)
    factual = {k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim <= 1 else v) for k, v in batch.items()}
    if "image" in factual and factual["image"].ndim == 3:
        factual["image"] = factual["image"].unsqueeze(0)
    return factual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["morphomnist", "celeba"], required=True)
    ap.add_argument("--graph", default="default")
    ap.add_argument("--variant", default=None, help="Graco variant subfolder name")
    ap.add_argument("--scm-config", type=Path, required=True)
    ap.add_argument("--classifier-config", type=Path, required=True)
    ap.add_argument("--cf-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--embeddings", type=str, default=None)
    ap.add_argument(
        "--uncond-vae-ckpt",
        type=Path,
        default=None,
        help="Unconditional VAE checkpoint when --embeddings vae.",
    )
    args = ap.parse_args()

    root = dataset_root(args.dataset)
    celeba_graph = args.graph if args.dataset == "celeba" else None
    apply_uncond_vae_env(
        os.environ,
        embeddings=args.embeddings,
        uncond_vae_ckpt=args.uncond_vae_ckpt,
        celeba_graph=celeba_graph,
    )
    with open(args.scm_config) as f:
        scm_cfg = json.load(f)

    attribute_size = scm_cfg["attribute_size"]
    parents = list(attribute_size.keys())
    transform = ReturnDictTransform(attribute_size)
    if args.dataset == "morphomnist":
        train_set = MorphoMNISTLike(attribute_size, split="train", transform=transform)
        test_set = MorphoMNISTLike(attribute_size, split="test", transform=transform)
        unnormalize_fn = unnormalize_morphomnist
    else:
        train_set = Celeba(attribute_size, split="train", transform=transform, data_dir=str(root / "data"))
        test_set = Celeba(attribute_size, split="test", transform=transform, data_dir=str(root / "data"))
        unnormalize_fn = unnormalize_celeba

    predictors, _ = load_predictors(args.classifier_config, scm_cfg)
    bundles = load_cf_bundles(args.cf_dir)
    if not bundles:
        raise SystemExit(f"No cf_*.pt in {args.cf_dir}")

    eff_by_parent: dict[str, dict[str, list]] = {p: {k: [] for k in attribute_size} for p in parents}
    counterfactual_images = []

    for b in tqdm(bundles, desc="effectiveness"):
        cf_batch = _batch_on_predictor_device(_batch_from_bundle(b), predictors)
        do_parent = b.get("meta", {}).get("do_parent", parents[0])
        e = effectiveness(cf_batch, unnormalize_fn, predictors, args.dataset)
        for k, v in e.items():
            if do_parent in eff_by_parent:
                eff_by_parent[do_parent][k].append(v)
        counterfactual_images.append(cf_batch["image"])

    results = {
        "variant": args.variant or args.cf_dir.name,
        "effectiveness_by_parent": {
            parent: {k: (float(np.mean(v)), float(np.std(v))) if v else (0.0, 0.0) for k, v in scores.items()}
            for parent, scores in eff_by_parent.items()
        },
    }

    batch_size = scm_cfg["mechanism_models"]["image"]["params"].get("batch_size_val", 64)
    real_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    fid_score = fid(real_loader, counterfactual_images)
    results["fid"] = float(fid_score)

    embedding_model = get_embedding_model(
        args.embeddings, pretrained_vgg=True, classifier_config=str(args.classifier_config)
    )
    embedding_fn = get_embedding_fn(args.embeddings, unnormalize_fn, embedding_model)

    factuals, counterfactuals, interventions = [], [], []
    for b in tqdm(bundles, desc="minimality"):
        meta = b.get("meta", {})
        idx = meta.get("sample_idx")
        do_parent = meta.get("do_parent")
        if idx is None or do_parent is None:
            continue
        factual_batch = _batch_on_predictor_device(_factual_batch(test_set, idx, transform), predictors)
        cf_batch = _batch_on_predictor_device(_batch_from_bundle(b), predictors)
        factual_cond = factual_batch.get("intensity")
        if args.embeddings == "vae":
            factual_cond = torch.cat([factual_batch[att] for att in parents], dim=1)
            counterfactual_cond = torch.cat([cf_batch[att] for att in parents], dim=1)
        else:
            counterfactual_cond = factual_cond
        factual_features = embedding_fn(
            factual_batch["image"],
            factual_cond if factual_cond is not None else None,
        )
        counterfactual_features = embedding_fn(
            cf_batch["image"],
            counterfactual_cond if counterfactual_cond is not None else None,
        )
        factuals.append((factual_features[0], factual_batch[do_parent].cpu().numpy()))
        counterfactuals.append((counterfactual_features[0], cf_batch[do_parent].cpu().numpy()))
        interventions.append(do_parent)

    if factuals:
        minimality_scores, _, _ = minimality(
            factuals, counterfactuals, interventions, test_set.bins, args.embeddings
        )
        results["minimality"] = (float(np.mean(minimality_scores)), float(np.std(minimality_scores)))

    out = args.out or (args.cf_dir / "graco_metrics.json")
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
