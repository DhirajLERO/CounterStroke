"""CFRL counterfactuals for ImageNet manifest samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_IMNET_DIR = Path(__file__).resolve().parent
_ROOT = _IMNET_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_IMNET_DIR))

from cfrl_runner_rgb import (
    cfrl_checkpoint_dir_ok,
    cfrl_explain_batch_nhwc,
    cfrl_manifest_matches,
    pred_np_cfrl_factory,
    train_cfrl_rgb_explainer_dataloader,
    write_cfrl_manifest,
)
from common.cfrl_targets import extract_intended_targets
from explainers.cfrl_compat import patch_alibi_cfrl_pytorch_load_model
from explainers.imagenet100_inception import load_imagenet100_classifier
from imagenet_alibi_config import CKPT_IMAGENET100_INCEPTION
from imagenet100_dataset import ImageNet100AlibiDataset, load_train_pool_indices, subset_loader
from imagenet_alibi_config import (
    AE_BATCH_CFPROTO,
    CFRL_BATCH_SIZE,
    CFRL_FIT_MAX_SAMPLES,
    CFRL_TRAIN_STEPS_NOTEBOOK,
    CKPT_CFRL_DIR,
)
from imagenet_data import add_imagenet_root_args, resolve_class_map_path
from imagenet_io import load_manifest_batch_nhwc
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
    p.add_argument("--train-pool", type=Path, default=_IMNET_DIR / "data_subset" / "train_pool_index.json")
    p.add_argument("--out-dir", type=Path, default=_IMNET_DIR / "outputs" / "generation")
    p.add_argument("--cfrl-checkpoint-dir", type=Path, default=_IMNET_DIR / "models" / CKPT_CFRL_DIR)
    p.add_argument("--cfrl-train-steps", type=int, default=CFRL_TRAIN_STEPS_NOTEBOOK)
    p.add_argument("--cfrl-retrain", action="store_true")
    p.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Cap CFRL fit pool. 0 = full train_pool.",
    )
    args = p.parse_args()

    from alibi.explainers import CounterfactualRL

    patch_alibi_cfrl_pytorch_load_model()

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.classifier_weights.is_file():
        raise SystemExit(f"Missing classifier {args.classifier_weights}; run train_classifier.py first.")
    inc = load_imagenet100_classifier(args.classifier_weights, device)

    pool = load_train_pool_indices(args.train_pool)
    if args.max_train_images is not None and args.max_train_images > 0:
        pool = pool[: args.max_train_images]

    pred_np_cfrl = pred_np_cfrl_factory(inc, device)
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
        train_ds = ImageNet100AlibiDataset(
            args.imagenet_root, "train", class_map_path=resolve_class_map_path(args.class_map)
        )
        train_loader = subset_loader(
            train_ds, pool, batch_size=AE_BATCH_CFPROTO, shuffle=True, num_workers=4
        )
        explainer = train_cfrl_rgb_explainer_dataloader(
            train_loader,
            args.cfrl_train_steps,
            device,
            inc,
            max_fit_samples=CFRL_FIT_MAX_SAMPLES,
        )
        cfrl_dir.mkdir(parents=True, exist_ok=True)
        explainer.save(str(cfrl_dir))
        write_cfrl_manifest(cfrl_dir, args.cfrl_train_steps)

    x_batch = load_manifest_batch_nhwc(
        manifest, args.imagenet_root, class_map_path=args.class_map
    )
    tgt_arr = extract_intended_targets(manifest, num_classes=100)
    cf_batch, hits, dt = cfrl_explain_batch_nhwc(explainer, x_batch, tgt_arr, CFRL_BATCH_SIZE)
    print(f"CFRL batched explain {len(manifest)} samples in {dt:.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for j, entry in enumerate(manifest):
        idx = int(entry["dataset_index"])
        cf_img = cf_batch[j : j + 1]
        sample_dir = args.out_dir / sample_dir_name(idx)
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta = load_or_create_meta(sample_dir, entry)
        meta["methods"]["cfrl"] = {"hit": bool(hits[j])}
        save_meta(sample_dir, meta)
        x_nhwc = x_batch[j : j + 1]
        upd = {"cf_cfrl": cf_img.astype(np.float32)}
        if not (sample_dir / "tensors.npz").is_file():
            upd["orig_nhwc"] = x_nhwc.astype(np.float32)
        merge_npz(sample_dir / "tensors.npz", upd)

    print("generate_cfrl ImageNet done")


if __name__ == "__main__":
    main()
