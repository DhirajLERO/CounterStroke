"""CounterfactualProto generation from a pretrained train-split checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_MNIST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MNIST_DIR))

from alibi_mnist_config import (
    CKPT_CFPROTO,
    CF_MAX_ITER,
    CFPROTO_FEATURE_RANGE,
)
from explainers.cfproto_pt import CounterfactualProtoPT
from explainers.mnist_cnn import MNISTNet
from io_samples import load_manifest, merge_npz, load_or_create_meta, save_meta
from mnist_alibi_data import load_mnist_alibi, nchw_to_nhwc, nhwc_to_nchw


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=_MNIST_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument("--out-dir", type=Path, default=_MNIST_DIR / "outputs" / "generation")
    p.add_argument("--mnist-root", type=Path, default=_MNIST_DIR / "data")
    p.add_argument("--pt-weights", type=Path, default=_MNIST_DIR / "models" / "mnist_cnn.pt")
    p.add_argument("--cfproto-checkpoint", type=Path, default=_MNIST_DIR / "models" / CKPT_CFPROTO)
    p.add_argument("--max-iter", type=int, default=CF_MAX_ITER)
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = MNISTNet()
    pt.load_state_dict(torch.load(args.pt_weights, map_location="cpu", weights_only=True))
    pt.eval().to(device)

    (_x_train, _y_train, _ohe_train), (x_all, _y_test, _ohe_test) = load_mnist_alibi(
        str(args.mnist_root)
    )

    proto_ckpt = args.cfproto_checkpoint
    if not proto_ckpt.is_file():
        raise SystemExit(f"Missing {proto_ckpt}; run train_explainer_models.py first.")
    proto = CounterfactualProtoPT.load_checkpoint(proto_ckpt, pt, device=device)
    if tuple(proto.feature_range) != tuple(CFPROTO_FEATURE_RANGE):
        raise ValueError(
            f"CFProto checkpoint {proto_ckpt} uses feature_range={proto.feature_range}; "
            f"expected canonical {CFPROTO_FEATURE_RANGE}. Retrain the checkpoint."
        )
    proto.max_iterations = args.max_iter
    print(f"Loaded cfproto {proto_ckpt}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest:
        idx = int(entry["sample_idx"])
        y_tgt = int(entry["y_intended_target"])
        x_1 = x_all[idx : idx + 1]
        x_pt = nhwc_to_nchw(x_1)
        proto_exp = proto.explain(x_pt, target_class=y_tgt)
        adv_pr = nchw_to_nhwc(proto_exp.adv)

        sample_dir = args.out_dir / f"sample_{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta = load_or_create_meta(sample_dir, entry)
        meta["methods"]["cfproto"] = {"success": bool(proto_exp.success), "adv_class": int(proto_exp.adv_class)}
        save_meta(sample_dir, meta)

        upd = {"cf_cfproto": adv_pr.astype(np.float32)}
        if not (sample_dir / "tensors.npz").is_file():
            upd["orig_nhwc"] = x_1.astype(np.float32)
        merge_npz(sample_dir / "tensors.npz", upd)
        print(f"sample_{idx:05d} cfproto")


if __name__ == "__main__":
    main()
