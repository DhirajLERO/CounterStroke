"""LeapFactual + LeapFactual_R."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_CIFAR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CIFAR_DIR))

from cifar_alibi_config import (
    CKPT_CIFAR_RESNET,
    CKPT_LEAP_CFM,
    CKPT_LEAP_VAE,
    LEAP_GAMMA_BLEND,
    LEAP_GAMMA_ROBUST,
    LEAP_LATENT_DIM,
    LEAP_MAX_ITR_BLEND,
    LEAP_MAX_ITR_ROBUST,
    LEAP_CFM_LAYERS,
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
from cifar_alibi_data import load_cifar10_alibi, nchw_to_nhwc, nhwc_to_nchw
from explainers.cifar_resnet import CifarResNet
from explainers.leapfactual_pt import LeapFactualCifar
from io_samples import load_manifest, merge_npz, load_or_create_meta, save_meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=_CIFAR_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument("--out-dir", type=Path, default=_CIFAR_DIR / "outputs" / "generation")
    p.add_argument("--cifar-root", type=Path, default=_CIFAR_DIR / "data")
    p.add_argument("--pt-weights", type=Path, default=_CIFAR_DIR / "models" / CKPT_CIFAR_RESNET)
    p.add_argument("--leap-vae", type=Path, default=_CIFAR_DIR / "models" / CKPT_LEAP_VAE)
    p.add_argument("--leap-cfm", type=Path, default=_CIFAR_DIR / "models" / CKPT_LEAP_CFM)
    args = p.parse_args()

    if not args.leap_vae.is_file() or not args.leap_cfm.is_file():
        raise SystemExit("Missing VAE/CFM checkpoints; run train_explainer_models.py")

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = CifarResNet()
    pt.load_state_dict(torch.load(args.pt_weights, map_location="cpu", weights_only=True))
    pt.eval().to(device)

    leap = LeapFactualCifar.from_checkpoints(
        pt,
        args.leap_vae,
        args.leap_cfm,
        device,
        latent_dim=LEAP_LATENT_DIM,
        mlp_hidden=LEAP_MLP_HIDDEN,
        mlp_layers=LEAP_CFM_LAYERS,
        num_classes=NUM_CLASSES,
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

    (_tr, _ytr, _ohe_tr), (x_all, _yte, _ohe_te) = load_cifar10_alibi(str(args.cifar_root))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest:
        idx = int(entry["sample_idx"])
        y_tgt = int(entry["y_intended_target"])
        x_1 = x_all[idx : idx + 1]
        x_pt = nhwc_to_nchw(x_1)
        lp_exp = leap.explain_nchw(x_pt, y_tgt)

        sample_dir = args.out_dir / f"sample_{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta = load_or_create_meta(sample_dir, entry)
        meta["methods"]["leapfactual"] = {"success": bool(lp_exp.success), "cf_class": int(lp_exp.cf_class)}
        meta["methods"]["leapfactual_r"] = {
            "success": bool(lp_exp.success_robust),
            "cf_class": int(lp_exp.cf_robust_class),
        }
        save_meta(sample_dir, meta)

        leap_cf = nchw_to_nhwc(lp_exp.cf).astype(np.float32)
        leap_cfr = nchw_to_nhwc(lp_exp.cf_robust).astype(np.float32)
        upd = {"cf_leapfactual": leap_cf, "cf_leapfactual_r": leap_cfr}
        if not (sample_dir / "tensors.npz").is_file():
            upd["orig_nhwc"] = x_1.astype(np.float32)
        merge_npz(sample_dir / "tensors.npz", upd)
        print(f"sample_{idx:05d} leap")


if __name__ == "__main__":
    main()
