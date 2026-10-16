"""CounterfactualProto — explain manifest samples using pretrained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_CIFAR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CIFAR_DIR))

from cifar_alibi_config import CKPT_CFPROTO, CKPT_CIFAR_RESNET, CF_MAX_ITER
from cifar_alibi_data import load_cifar10_alibi, nchw_to_nhwc, nhwc_to_nchw
from explainers.cifar_resnet import CifarResNet
from explainers.cfproto_pt import CounterfactualProtoPT
from io_samples import load_manifest, merge_npz, load_or_create_meta, save_meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=_CIFAR_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument("--out-dir", type=Path, default=_CIFAR_DIR / "outputs" / "generation")
    p.add_argument("--cifar-root", type=Path, default=_CIFAR_DIR / "data")
    p.add_argument("--pt-weights", type=Path, default=_CIFAR_DIR / "models" / CKPT_CIFAR_RESNET)
    p.add_argument("--cfproto-checkpoint", type=Path, default=_CIFAR_DIR / "models" / CKPT_CFPROTO)
    p.add_argument("--max-iter", type=int, default=CF_MAX_ITER)
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = CifarResNet()
    pt.load_state_dict(torch.load(args.pt_weights, map_location="cpu", weights_only=True))
    pt.eval().to(device)

    (_tr, _ytr, _ohe_tr), (x_all, _yte, _ohe_te) = load_cifar10_alibi(str(args.cifar_root))

    proto_ckpt = args.cfproto_checkpoint
    if not proto_ckpt.is_file():
        raise SystemExit(f"Missing {proto_ckpt}; run train_explainer_models.py first.")
    proto = CounterfactualProtoPT.load_checkpoint(proto_ckpt, pt, device=device)
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
