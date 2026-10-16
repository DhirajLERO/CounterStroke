"""Latent KDTree CFProto for the ImageNet-100 evaluation manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_IMNET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_IMNET_DIR))

from explainers.cfproto_pt_rgb import CounterfactualProtoRGB
from explainers.imagenet100_inception import load_imagenet100_classifier
from imagenet100_dataset import ImageNet100AlibiDataset, load_train_pool_indices, subset_loader
from imagenet_data import add_imagenet_root_args, resolve_class_map_path
from imagenet_io import load_manifest_entry_nhwc
from imagenet_alibi_config import (
    AE_BATCH_CFPROTO,
    AE_EPOCHS_CFPROTO,
    CFPROTO_C_INIT,
    CFPROTO_C_STEPS,
    CFPROTO_FEATURE_RANGE,
    CFPROTO_GAMMA,
    CFPROTO_LATENT_DIM,
    CFPROTO_LR,
    CFPROTO_THETA,
    CKPT_IMAGENET100_INCEPTION,
    CKPT_CFPROTO,
    CF_MAX_ITER,
    NUM_CLASSES,
)
from io_samples import load_manifest, load_or_create_meta, merge_npz, sample_dir_name, save_meta


def _nchw_to_nhwc(a: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(a, dtype=np.float32), (0, 2, 3, 1))


def _ensure_proto_for_target_class(
    proto: CounterfactualProtoRGB,
    ds: ImageNet100AlibiDataset,
    y_tgt: int,
    *,
    max_images: int = 64,
) -> None:
    """CFProto fit() only indexes labels present in the configured train pool."""
    if y_tgt in proto._latent_by_class:
        return
    targets = np.asarray(ds.targets, dtype=np.int64)
    idxs = np.flatnonzero(targets == int(y_tgt))
    if idxs.size == 0:
        raise ValueError(
            f"No ImageNet-100 train samples for synset class {y_tgt}; cannot build latent index."
        )
    idxs = idxs[: int(max_images)]
    chunks: list[np.ndarray] = []
    for ii in idxs:
        img_t, _ = ds[int(ii)]
        chunks.append(np.asarray(img_t.numpy(), dtype=np.float32))
    stacked = np.stack(chunks, axis=0)
    proto.add_target_class_samples(int(y_tgt), stacked)


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
    p.add_argument("--cfproto-checkpoint", type=Path, default=_IMNET_DIR / "models" / CKPT_CFPROTO)
    p.add_argument("--cfproto-retrain", action="store_true")
    p.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Cap CFProto fit pool (first N indices). 0 = full train_pool.",
    )
    p.add_argument("--max-iter", type=int, default=min(CF_MAX_ITER, 400))
    args = p.parse_args()

    manifest = load_manifest(args.manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.classifier_weights.is_file():
        raise SystemExit(f"Missing classifier {args.classifier_weights}; run train_classifier.py first.")
    inc = load_imagenet100_classifier(args.classifier_weights, device)

    pool = load_train_pool_indices(args.train_pool)
    if args.max_train_images is not None and args.max_train_images > 0:
        pool = pool[: args.max_train_images]

    cmap_path = resolve_class_map_path(args.class_map)
    train_ds = ImageNet100AlibiDataset(
        args.imagenet_root, "train", class_map_path=cmap_path
    )
    ckpt = args.cfproto_checkpoint
    if ckpt.is_file() and not args.cfproto_retrain:
        proto = CounterfactualProtoRGB.load_checkpoint(ckpt, inc, NUM_CLASSES, device=device)
        if tuple(proto.feature_range) != tuple(CFPROTO_FEATURE_RANGE):
            raise ValueError(
                f"CFProto checkpoint {ckpt} uses feature_range={proto.feature_range}; "
                f"expected canonical {CFPROTO_FEATURE_RANGE}. Retrain the checkpoint."
            )
        if proto.latent_dim != CFPROTO_LATENT_DIM:
            raise ValueError(
                f"CFProto checkpoint {ckpt} uses latent_dim={proto.latent_dim}; "
                f"expected {CFPROTO_LATENT_DIM}. Retrain the checkpoint."
            )
        print(f"Loaded CFProto {ckpt}")
    else:
        proto = CounterfactualProtoRGB(
            inc,
            num_classes=NUM_CLASSES,
            device=device,
            kappa=0.0,
            beta=0.1,
            gamma=CFPROTO_GAMMA,
            theta=CFPROTO_THETA,
            learning_rate=CFPROTO_LR,
            max_iterations=args.max_iter,
            c_init=CFPROTO_C_INIT,
            c_steps=CFPROTO_C_STEPS,
            feature_range=CFPROTO_FEATURE_RANGE,
            ae_train_epochs=AE_EPOCHS_CFPROTO,
            ae_batch_size=AE_BATCH_CFPROTO,
            latent_dim=CFPROTO_LATENT_DIM,
        )
        train_loader = subset_loader(
            train_ds, pool, batch_size=AE_BATCH_CFPROTO, shuffle=True, num_workers=4
        )
        proto.fit_dataloader(train_loader)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        proto.save_checkpoint(ckpt)
        print(f"Saved CFProto {ckpt}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest:
        idx = int(entry["dataset_index"])
        y_tgt = int(entry["y_intended_target"])
        _ensure_proto_for_target_class(proto, train_ds, y_tgt)
        x_nhwc, _y, _ = load_manifest_entry_nhwc(
            entry, args.imagenet_root, class_map_path=cmap_path
        )
        x_chw = np.transpose(x_nhwc[np.newaxis, ...], (0, 3, 1, 2)).astype(np.float32)
        exp = proto.explain(x_chw, target_class=y_tgt)
        adv_nhwc = _nchw_to_nhwc(exp.adv)

        sample_dir = args.out_dir / sample_dir_name(idx)
        sample_dir.mkdir(parents=True, exist_ok=True)
        meta = load_or_create_meta(sample_dir, entry)
        meta["methods"]["cfproto"] = {"success": bool(exp.success), "adv_class": int(exp.adv_class)}
        save_meta(sample_dir, meta)

        x_nhwc = _nchw_to_nhwc(x_chw)
        upd = {"cf_cfproto": adv_nhwc.astype(np.float32)}
        if not (sample_dir / "tensors.npz").is_file():
            upd["orig_nhwc"] = x_nhwc.astype(np.float32)
        merge_npz(sample_dir / "tensors.npz", upd)
        print(sample_dir.name, "cfproto ok")


if __name__ == "__main__":
    main()
