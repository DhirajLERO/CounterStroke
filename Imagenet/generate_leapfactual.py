"""LeapFactual + LeapFactual_R for ImageNet manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
_IMNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_IMNET_DIR))

from explainers.imagenet100_inception import load_imagenet100_classifier
from imagenet_alibi_config import CKPT_IMAGENET100_INCEPTION
from explainers.leapfactual_pt_rgb import LeapFactualImageNet
from imagenet_alibi_config import (
    CKPT_LEAP_CFM,
    CKPT_LEAP_VAE,
    LEAP_CFM_LAYERS,
    LEAP_GAMMA_BLEND,
    LEAP_GAMMA_ROBUST,
    LEAP_LATENT_DIM,
    LEAP_MAX_ITR_BLEND,
    LEAP_MAX_ITR_ROBUST,
    LEAP_MLP_HIDDEN,
    LEAP_N_ODE_STEPS,
    LEAP_ODE_ATOL,
    LEAP_ODE_RTOL,
    LEAP_ODE_SENSITIVITY,
    LEAP_ODE_SOLVER,
    LEAP_T_LOWER,
    LEAP_T_UPPER,
    NUM_CLASSES,
)
from imagenet_data import add_imagenet_root_args
from imagenet_io import load_manifest_entry_nhwc
from io_samples import load_manifest, load_or_create_meta, merge_npz, sample_dir_name, save_meta


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
    p.add_argument("--leap-vae", type=Path, default=_IMNET_DIR / "models" / CKPT_LEAP_VAE)
    p.add_argument("--leap-cfm", type=Path, default=_IMNET_DIR / "models" / CKPT_LEAP_CFM)
    args = p.parse_args()

    if not args.leap_vae.is_file() or not args.leap_cfm.is_file():
        raise SystemExit("Missing VAE/CFM checkpoints; run train_explainer_models.py first.")

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.classifier_weights.is_file():
        raise SystemExit(f"Missing classifier {args.classifier_weights}; run train_classifier.py first.")
    inc = load_imagenet100_classifier(args.classifier_weights, device)

    leap = LeapFactualImageNet.from_checkpoints(
        inc,
        args.leap_vae,
        args.leap_cfm,
        device,
        latent_dim=LEAP_LATENT_DIM,
        num_classes=NUM_CLASSES,
        mlp_hidden=LEAP_MLP_HIDDEN,
        mlp_layers=LEAP_CFM_LAYERS,
        gamma_blend=LEAP_GAMMA_BLEND,
        max_itr_blend=LEAP_MAX_ITR_BLEND,
        gamma_robust=LEAP_GAMMA_ROBUST,
        max_itr_robust=LEAP_MAX_ITR_ROBUST,
        n_ode_steps=LEAP_N_ODE_STEPS,
        t_lower=LEAP_T_LOWER,
        t_upper=LEAP_T_UPPER,
        ode_solver=LEAP_ODE_SOLVER,
        ode_sensitivity=LEAP_ODE_SENSITIVITY,
        ode_atol=LEAP_ODE_ATOL,
        ode_rtol=LEAP_ODE_RTOL,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest:
        idx = int(entry["dataset_index"])
        y_tgt = int(entry["y_intended_target"])
        x_nhwc, _y, _ = load_manifest_entry_nhwc(
            entry, args.imagenet_root, class_map_path=args.class_map
        )
        x_chw = np.transpose(x_nhwc[np.newaxis, ...], (0, 3, 1, 2)).astype(np.float32)
        lp_exp = leap.explain_nchw(x_chw, y_tgt)

        def _nhwc(x: np.ndarray) -> np.ndarray:
            return np.transpose(x, (0, 2, 3, 1))

        leap_cf = _nhwc(lp_exp.cf).astype(np.float32)
        leap_cfr = _nhwc(lp_exp.cf_robust).astype(np.float32)

        sample_dir = args.out_dir / sample_dir_name(idx)
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta = load_or_create_meta(sample_dir, entry)
        meta["methods"]["leapfactual"] = {"success": bool(lp_exp.success), "cf_class": int(lp_exp.cf_class)}
        meta["methods"]["leapfactual_r"] = {
            "success": bool(lp_exp.success_robust),
            "cf_class": int(lp_exp.cf_robust_class),
        }
        save_meta(sample_dir, meta)

        x_nhwc = _nhwc(x_chw).astype(np.float32)
        upd = {"cf_leapfactual": leap_cf, "cf_leapfactual_r": leap_cfr}
        if not (sample_dir / "tensors.npz").is_file():
            upd["orig_nhwc"] = x_nhwc
        merge_npz(sample_dir / "tensors.npz", upd)
        print(sample_dir.name, "leap ok")


if __name__ == "__main__":
    main()
