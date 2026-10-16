"""Counterfactual RL generation for CIFAR-10 manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from alibi.explainers import CounterfactualRL

_CIFAR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CIFAR_DIR))

from cfrl_runner_cifar import CFRL_BATCH_SIZE, cfrl_checkpoint_dir_ok, cfrl_explain_batch_nhwc
from cifar_alibi_config import CKPT_CFRL_DIR, CKPT_CIFAR_RESNET
from cifar_alibi_data import load_cifar10_alibi
from explainers.cfrl_compat import patch_alibi_cfrl_pytorch_load_model
from explainers.cifar_resnet import CifarResNet
from io_samples import load_manifest, merge_npz, load_or_create_meta, save_meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=_CIFAR_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument("--out-dir", type=Path, default=_CIFAR_DIR / "outputs" / "generation")
    p.add_argument("--cifar-root", type=Path, default=_CIFAR_DIR / "data")
    p.add_argument("--pt-weights", type=Path, default=_CIFAR_DIR / "models" / CKPT_CIFAR_RESNET)
    p.add_argument("--cfrl-checkpoint-dir", type=Path, default=_CIFAR_DIR / "models" / CKPT_CFRL_DIR)
    args = p.parse_args()

    patch_alibi_cfrl_pytorch_load_model()

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = CifarResNet()
    pt.load_state_dict(torch.load(args.pt_weights, map_location="cpu", weights_only=True))
    pt.eval().to(device)

    (_tr, _ytr, _ohe_tr), (x_all, _yte, _ohe_te) = load_cifar10_alibi(str(args.cifar_root))

    cfrl_dir = args.cfrl_checkpoint_dir
    if not cfrl_checkpoint_dir_ok(cfrl_dir):
        raise SystemExit(f"Missing CFRL checkpoint dir {cfrl_dir}; run train_explainer_models.py first.")

    from cfrl_runner_cifar import pred_np_cfrl_factory

    pred_np_cfrl = pred_np_cfrl_factory(pt, device)
    explainer = CounterfactualRL.load(str(cfrl_dir), pred_np_cfrl)
    print(f"CFRL loaded {cfrl_dir}")

    indices = [int(e["sample_idx"]) for e in manifest]
    x_batch = np.stack([x_all[i] for i in indices], axis=0).astype(np.float32)
    tgt_arr = np.asarray([int(e["y_intended_target"]) for e in manifest], dtype=np.int64)

    cf_batch, hits, dt = cfrl_explain_batch_nhwc(explainer, x_batch, tgt_arr, CFRL_BATCH_SIZE)
    print(f"cfrl batch explain in {dt:.3f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for j, entry in enumerate(manifest):
        idx = int(entry["sample_idx"])
        sample_dir = args.out_dir / f"sample_{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta = load_or_create_meta(sample_dir, entry)
        meta["methods"]["cfrl"] = {"hit": bool(hits[j])}
        save_meta(sample_dir, meta)
        cf_img = cf_batch[j : j + 1]
        upd = {"cf_cfrl": cf_img.astype(np.float32)}
        if not (sample_dir / "tensors.npz").is_file():
            upd["orig_nhwc"] = x_all[idx : idx + 1].astype(np.float32)
        merge_npz(sample_dir / "tensors.npz", upd)
        print(f"sample_{idx:05d} cfrl")

    print("generate_cfrl done")


if __name__ == "__main__":
    main()
