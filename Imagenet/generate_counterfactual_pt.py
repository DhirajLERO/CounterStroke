"""Gradient CF (other-class) for ImageNet eval manifest (Inception v3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_IMNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_IMNET_DIR))

from explainers.counterfactual_pt import CounterfactualPT
from imagenet_alibi_config import CF_LR, CF_MAX_ITER, CF_MAX_LAM_STEPS, CF_TOL
from explainers.imagenet100_inception import load_imagenet100_classifier
from imagenet_alibi_config import CKPT_IMAGENET100_INCEPTION
from imagenet_data import add_imagenet_root_args
from imagenet_io import load_manifest_entry_nhwc
from io_samples import load_manifest, merge_npz, load_or_create_meta, sample_dir_name, save_meta


def _nchw_to_nhwc_rgb(a: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(a, dtype=np.float32), (0, 2, 3, 1))


def main() -> None:
    p = argparse.ArgumentParser()
    add_imagenet_root_args(p)
    p.add_argument(
        "--classifier-weights",
        type=Path,
        default=_IMNET_DIR / "models" / CKPT_IMAGENET100_INCEPTION,
    )
    p.add_argument("--manifest", type=Path, default=_IMNET_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument("--out-dir", type=Path, default=_IMNET_DIR / "outputs" / "generation")
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.classifier_weights.is_file():
        raise SystemExit(f"Missing classifier {args.classifier_weights}; run train_classifier.py first.")
    pt = load_imagenet100_classifier(args.classifier_weights, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest:
        idx = int(entry["dataset_index"])
        x_nhwc, _y, _ = load_manifest_entry_nhwc(
            entry, args.imagenet_root, class_map_path=args.class_map
        )
        x_nchw = np.transpose(x_nhwc[np.newaxis, ...], (0, 3, 1, 2))

        feature_range = (-0.5, 0.5)
        cf_exp = CounterfactualPT(
            pt,
            device=device,
            target_class="other",
            max_iter=min(CF_MAX_ITER, 500),
            max_lam_steps=CF_MAX_LAM_STEPS,
            tol=CF_TOL,
            learning_rate_init=CF_LR,
            feature_range=feature_range,
        ).explain(x_nchw)
        adv = _nchw_to_nhwc_rgb(cf_exp.cf)

        sample_dir = args.out_dir / sample_dir_name(idx)
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta = load_or_create_meta(sample_dir, entry)
        meta["methods"]["counterfactual_pt"] = {
            "success": bool(cf_exp.success),
            "cf_class": int(cf_exp.cf_class),
        }
        save_meta(sample_dir, meta)

        merge_npz(
            sample_dir / "tensors.npz",
            {"orig_nhwc": x_nhwc.astype(np.float32), "cf_counterfactual_pt": adv.astype(np.float32)},
        )
        print(sample_dir.name, "ok")


if __name__ == "__main__":
    main()
