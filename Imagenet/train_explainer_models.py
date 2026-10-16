"""Train CFProto, CFRL, Leap WAE+CFM on ImageNet-100 train split."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_IMNET_DIR = Path(__file__).resolve().parent
_GRACO_ROOT = _IMNET_DIR.parent
sys.path.insert(0, str(_GRACO_ROOT))
sys.path.insert(0, str(_IMNET_DIR))

from cfrl_runner_rgb import train_cfrl_rgb_explainer_dataloader, write_cfrl_manifest
from explainers.cfproto_pt_rgb import CounterfactualProtoRGB
from explainers.cfm_mlp_rgb import MLP_CFM
from explainers.imagenet100_inception import load_imagenet100_classifier
from imagenet100_dataset import ImageNet100AlibiDataset, load_train_pool_indices, subset_loader
from imagenet_data import add_imagenet_root_args, resolve_class_map_path
from common.leap_wae import build_leap_wae_adapter, train_wae_mmd_dataloader
from imagenet_alibi_config import (
    AE_BATCH_CFPROTO,
    AE_EPOCHS_CFPROTO,
    CKPT_IMAGENET100_INCEPTION,
    CKPT_CFPROTO,
    CKPT_CFRL_DIR,
    CKPT_LEAP_CFM,
    CKPT_LEAP_VAE,
    CFPROTO_FEATURE_RANGE,
    CFPROTO_LATENT_DIM,
    CFRL_FIT_MAX_SAMPLES,
    CFRL_TRAIN_STEPS_NOTEBOOK,
    LEAP_CFM_BATCH,
    LEAP_CFM_COND_MODE,
    LEAP_CFM_EPOCHS,
    LEAP_CFM_LAYERS,
    LEAP_CFM_LR,
    LEAP_CFM_SIGMA,
    LEAP_LATENT_DIM,
    LEAP_MLP_HIDDEN,
    LEAP_VAE_BATCH,
    LEAP_VAE_EPOCHS,
    LEAP_WAE_AMP,
    NUM_CLASSES,
)
def _load_pool_indices(train_pool_json: Path) -> list[int]:
    return load_train_pool_indices(train_pool_json)


def _cfm_conditioning(clf, xb: torch.Tensor, yb: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "probs":
        return clf.predict_proba(xb)
    if mode == "one_hot":
        return F.one_hot(yb.long(), NUM_CLASSES).float()
    if mode == "mixed":
        if torch.rand(1, device=xb.device).item() < 0.5:
            return clf.predict_proba(xb)
        return F.one_hot(yb.long(), NUM_CLASSES).float()
    raise ValueError(f"Unknown LEAP_CFM_COND_MODE {mode!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    add_imagenet_root_args(ap)
    ap.add_argument(
        "--train-pool",
        type=Path,
        default=_IMNET_DIR / "data_subset" / "train_pool_index.json",
    )
    ap.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Cap train indices. 0 = full train pool.",
    )
    ap.add_argument("--wae-val-images", type=int, default=4096, help="Val images for WAE SSIM (imagenet100)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-cfrl", action="store_true")
    ap.add_argument("--skip-leap", action="store_true")
    ap.add_argument("--skip-cfproto", action="store_true")
    ap.add_argument("--cfrl-steps", type=int, default=CFRL_TRAIN_STEPS_NOTEBOOK)
    ap.add_argument(
        "--classifier-weights",
        type=Path,
        default=_IMNET_DIR / "models" / CKPT_IMAGENET100_INCEPTION,
    )
    args = ap.parse_args()

    if not args.classifier_weights.is_file():
        raise SystemExit(f"Missing classifier {args.classifier_weights}; run train_classifier.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inc = load_imagenet100_classifier(args.classifier_weights, device)

    pool = _load_pool_indices(args.train_pool)
    if args.max_train_images is not None and args.max_train_images > 0:
        pool = pool[: args.max_train_images]
    print(f"Train pool: {len(pool)} images")

    models_dir = _IMNET_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cmap_path = resolve_class_map_path(args.class_map)
    train_ds = ImageNet100AlibiDataset(args.imagenet_root, "train", class_map_path=cmap_path)
    val_ds = ImageNet100AlibiDataset(args.imagenet_root, "val", class_map_path=cmap_path)
    train_loader = subset_loader(
        train_ds, pool, batch_size=AE_BATCH_CFPROTO, shuffle=True, num_workers=4
    )

    cfproto_ckpt = models_dir / CKPT_CFPROTO
    if not args.skip_cfproto and (args.force or not cfproto_ckpt.is_file()):
        proto = CounterfactualProtoRGB(
            inc,
            num_classes=NUM_CLASSES,
            device=device,
            feature_range=CFPROTO_FEATURE_RANGE,
            ae_train_epochs=AE_EPOCHS_CFPROTO,
            ae_batch_size=AE_BATCH_CFPROTO,
            latent_dim=CFPROTO_LATENT_DIM,
        )
        proto.fit_dataloader(train_loader)
        proto.save_checkpoint(cfproto_ckpt)
        print(f"Saved CFProto {cfproto_ckpt}")
    elif not args.skip_cfproto:
        print(f"Skip CFProto ({cfproto_ckpt} exists).")

    cfrl_dir = models_dir / CKPT_CFRL_DIR
    if not args.skip_cfrl and (args.force or not cfrl_dir.is_dir()):
        explainer = train_cfrl_rgb_explainer_dataloader(
            train_loader,
            args.cfrl_steps,
            device,
            inc,
            max_fit_samples=CFRL_FIT_MAX_SAMPLES,
        )
        cfrl_dir.mkdir(parents=True, exist_ok=True)
        explainer.save(str(cfrl_dir))
        write_cfrl_manifest(cfrl_dir, args.cfrl_steps)
        print(f"Saved CFRL {cfrl_dir}")
    elif not args.skip_cfrl:
        print(f"Skip CFRL ({cfrl_dir} exists).")

    vae_path = models_dir / CKPT_LEAP_VAE
    cfm_path = models_dir / CKPT_LEAP_CFM
    need_vae = not args.skip_leap and (args.force or not vae_path.is_file())
    need_cfm = not args.skip_leap and (args.force or not cfm_path.is_file() or need_vae)

    if need_vae or need_cfm:
        val_n = min(len(val_ds), max(1, int(args.wae_val_images)))
        val_idx = random.sample(range(len(val_ds)), val_n) if val_n < len(val_ds) else list(range(len(val_ds)))
        wae_train_loader = subset_loader(
            train_ds, pool, batch_size=LEAP_VAE_BATCH, shuffle=True, num_workers=4
        )
        wae_val_loader = subset_loader(
            val_ds, val_idx, batch_size=LEAP_VAE_BATCH, shuffle=False, num_workers=2
        )

        if need_vae:
            print("Training Leap WAE_MMD (ImageNet-100 train stream)...")
            train_wae_mmd_dataloader(
                wae_train_loader,
                wae_val_loader,
                dataset="imagenet",
                save_path=vae_path,
                device=device,
                train_params={"max_epochs": LEAP_VAE_EPOCHS},
                use_amp=LEAP_WAE_AMP,
            )

        vae = build_leap_wae_adapter("imagenet", device=device)
        vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
        vae.eval()

        if need_cfm:
            cfm = MLP_CFM(
                dim=LEAP_LATENT_DIM,
                time_varying=True,
                cond=True,
                cond_dim=NUM_CLASSES,
                w=LEAP_MLP_HIDDEN,
                n_hidden_layers=LEAP_CFM_LAYERS,
            ).to(device)
            from torchcfm import ConditionalFlowMatcher

            fm = ConditionalFlowMatcher(sigma=LEAP_CFM_SIGMA)
            opt_c = torch.optim.Adam(cfm.parameters(), lr=LEAP_CFM_LR)
            cfm.train()
            for _ in tqdm(range(LEAP_CFM_EPOCHS), desc="Leap CFM"):
                for xb, yb in wae_train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    with torch.no_grad():
                        mu_z, _ = vae.encode(xb)
                        x1 = mu_z
                        y = _cfm_conditioning(inc, xb, yb, LEAP_CFM_COND_MODE)
                    x0 = torch.randn_like(x1)
                    t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
                    vt = cfm(t, torch.cat([xt, y], dim=-1))
                    loss = torch.mean((vt - ut) ** 2)
                    opt_c.zero_grad(set_to_none=True)
                    loss.backward()
                    opt_c.step()
            torch.save(cfm.state_dict(), cfm_path)
            print(f"Saved Leap CFM {cfm_path}")
        elif not args.skip_leap:
            print("Skip Leap CFM (checkpoint exists).")
    elif not args.skip_leap:
        print("Skip Leap (checkpoints exist).")


if __name__ == "__main__":
    main()
