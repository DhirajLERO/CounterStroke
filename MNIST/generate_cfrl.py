"""CFRL counterfactuals for manifest samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import MNIST

_MNIST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MNIST_DIR))

from alibi_mnist_config import CFRL_BATCH_SIZE, CFRL_TRAIN_STEPS_NOTEBOOK, CKPT_CFRL_DIR
from cfrl_runner import (
    cfrl_checkpoint_dir_ok,
    cfrl_explain_batch_nhwc,
    cfrl_manifest_matches,
    train_cfrl_pytorch_explainer,
    write_cfrl_manifest,
)
from explainers.cfrl_compat import patch_alibi_cfrl_pytorch_load_model
from explainers.mnist_cnn import MNISTNet
from io_samples import load_manifest, merge_npz, load_or_create_meta, save_meta

FIT_START = 100


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=_MNIST_DIR / "data_subset" / "eval_manifest.json")
    p.add_argument("--out-dir", type=Path, default=_MNIST_DIR / "outputs" / "generation")
    p.add_argument("--mnist-root", type=Path, default=_MNIST_DIR / "data")
    p.add_argument("--pt-weights", type=Path, default=_MNIST_DIR / "models" / "mnist_cnn.pt")
    p.add_argument("--cfrl-checkpoint-dir", type=Path, default=_MNIST_DIR / "models" / CKPT_CFRL_DIR)
    p.add_argument("--cfrl-train-steps", type=int, default=CFRL_TRAIN_STEPS_NOTEBOOK)
    p.add_argument("--cfrl-retrain", action="store_true")
    args = p.parse_args()

    from alibi.explainers import CounterfactualRL

    patch_alibi_cfrl_pytorch_load_model()

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pt = MNISTNet()
    pt.load_state_dict(torch.load(args.pt_weights, map_location="cpu", weights_only=True))
    pt.eval().to(device)

    train_raw = MNIST(root=str(args.mnist_root), train=False, download=True, transform=None)
    x_all = (train_raw.data.float().numpy() / 255.0).reshape(-1, 28, 28, 1).astype(np.float32) - 0.5
    y_all = train_raw.targets.numpy().astype(np.int64)
    x_fit = x_all[FIT_START:]
    y_fit = y_all[FIT_START:]

    from cfrl_runner import pred_np_cfrl_factory

    pred_np_cfrl = pred_np_cfrl_factory(pt, device)
    cfrl_dir = args.cfrl_checkpoint_dir
    try_load = (
        not args.cfrl_retrain
        and cfrl_checkpoint_dir_ok(cfrl_dir)
        and cfrl_manifest_matches(cfrl_dir, args.cfrl_train_steps)
    )
    explainer = None
    if try_load:
        try:
            explainer = CounterfactualRL.load(str(cfrl_dir), pred_np_cfrl)
            print(f"CFRL loaded {cfrl_dir}")
        except Exception as e:
            print(f"CFRL load failed {e}; fitting.")
    if explainer is None:
        explainer = train_cfrl_pytorch_explainer(x_fit, args.cfrl_train_steps, device, pt)
        cfrl_dir.mkdir(parents=True, exist_ok=True)
        explainer.save(str(cfrl_dir))
        write_cfrl_manifest(cfrl_dir, args.cfrl_train_steps)

    idx_list = [int(e["sample_idx"]) for e in manifest]
    tgt_list = [int(e["y_intended_target"]) for e in manifest]
    x_batch = np.concatenate([x_all[i : i + 1] for i in idx_list], axis=0)
    tgt_arr = np.array(tgt_list, dtype=np.int64)

    cf_batch, hits, dt = cfrl_explain_batch_nhwc(explainer, x_batch, tgt_arr, CFRL_BATCH_SIZE)
    print(f"CFRL batched explain {len(idx_list)} samples in {dt:.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for j, entry in enumerate(manifest):
        idx = int(entry["sample_idx"])
        cf_img = cf_batch[j : j + 1]
        sample_dir = args.out_dir / f"sample_{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta = load_or_create_meta(sample_dir, entry)
        meta["methods"]["cfrl"] = {"hit": bool(hits[j])}
        save_meta(sample_dir, meta)
        x_1 = x_all[idx : idx + 1]
        upd = {"cf_cfrl": cf_img.astype(np.float32)}
        if not (sample_dir / "tensors.npz").is_file():
            upd["orig_nhwc"] = x_1.astype(np.float32)
        merge_npz(sample_dir / "tensors.npz", upd)

    print("generate_cfrl done")


if __name__ == "__main__":
    main()
