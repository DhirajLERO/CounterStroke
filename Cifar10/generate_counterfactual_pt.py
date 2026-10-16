"""Gradient-based CounterfactualPT for each manifest sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_CIFAR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CIFAR_DIR))

from cifar_alibi_config import CF_LR, CF_MAX_ITER, CF_MAX_LAM_STEPS, CF_TOL, CKPT_CIFAR_RESNET
from cifar_alibi_data import load_cifar10_alibi, nchw_to_nhwc, nhwc_to_nchw
from explainers.cifar_resnet import CifarResNet
from explainers.counterfactual_pt import CounterfactualPT
from io_samples import load_manifest, merge_npz, load_or_create_meta, save_meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=_CIFAR_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument("--out-dir", type=Path, default=_CIFAR_DIR / "outputs" / "generation")
    p.add_argument("--cifar-root", type=Path, default=_CIFAR_DIR / "data")
    p.add_argument("--pt-weights", type=Path, default=_CIFAR_DIR / "models" / CKPT_CIFAR_RESNET)
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = CifarResNet()
    pt.load_state_dict(torch.load(args.pt_weights, map_location="cpu", weights_only=True))
    pt.eval().to(device)

    (_tr, _ytr, _ohe_tr), (x_all, _yte, _ohe_te) = load_cifar10_alibi(str(args.cifar_root))
    feature_range = (float(x_all.min()), float(x_all.max()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    root_meta = {"manifest": str(args.manifest), "method": "counterfactual_pt"}
    (args.out_dir / "generation_meta.json").write_text(json.dumps(root_meta, indent=2), encoding="utf-8")

    for entry in manifest:
        idx = int(entry["sample_idx"])
        x_1 = x_all[idx : idx + 1]
        x_pt = nhwc_to_nchw(x_1)

        cf_exp = CounterfactualPT(
            pt,
            device=device,
            target_class="other",
            max_iter=CF_MAX_ITER,
            max_lam_steps=CF_MAX_LAM_STEPS,
            tol=CF_TOL,
            learning_rate_init=CF_LR,
            feature_range=feature_range,
        ).explain(x_pt)
        adv_cf = nchw_to_nhwc(cf_exp.cf)

        sample_dir = args.out_dir / f"sample_{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta = load_or_create_meta(sample_dir, entry)
        meta["methods"]["counterfactual_pt"] = {
            "success": bool(cf_exp.success),
            "cf_class": int(cf_exp.cf_class),
        }
        save_meta(sample_dir, meta)

        merge_npz(
            sample_dir / "tensors.npz",
            {"orig_nhwc": x_1.astype(np.float32), "cf_counterfactual_pt": adv_cf.astype(np.float32)},
        )
        print(f"sample_{idx:05d} counterfactual_pt ok")


if __name__ == "__main__":
    main()
