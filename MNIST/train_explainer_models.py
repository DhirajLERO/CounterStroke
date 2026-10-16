"""Train CFProto and CFRL on the MNIST train pool, preserving Leap training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_MNIST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MNIST_DIR))

from alibi_mnist_config import (
    CKPT_LEAP_CFM,
    CKPT_LEAP_VAE,
    CKPT_MNIST_CNN,
    CKPT_CFPROTO,
    CKPT_CFRL_DIR,
    CFRL_TRAIN_STEPS_NOTEBOOK,
    AE_BATCH_KERAS,
    AE_EPOCHS_CFPROTO,
    CFPROTO_C_INIT,
    CFPROTO_C_STEPS,
    CFPROTO_FEATURE_RANGE,
    CFPROTO_GAMMA,
    CFPROTO_LR,
    CFPROTO_THETA,
    CF_MAX_ITER,
    LEAP_CFM_BATCH,
    LEAP_CFM_EPOCHS,
    LEAP_CFM_LR,
    LEAP_CFM_SIGMA,
    LEAP_LATENT_DIM,
    LEAP_MLP_HIDDEN,
    LEAP_VAE_BATCH,
    LEAP_VAE_EPOCHS,
    LEAP_VAE_KL_W,
    LEAP_VAE_LR,
)
from cfrl_runner import (
    cfrl_checkpoint_dir_ok,
    cfrl_manifest_matches,
    train_cfrl_pytorch_explainer,
    write_cfrl_manifest,
)
from explainers.cfproto_pt import CounterfactualProtoPT
from explainers.cfm_mlp import MLP_CFM
from explainers.mnist_cnn import MNISTNet
from explainers.vae_latent import MNISTVAELatent32
from mnist_alibi_data import load_mnist_alibi, nhwc_to_nchw


def _load_pool_indices(train_pool_json: Path) -> list[int]:
    data = json.loads(train_pool_json.read_text(encoding="utf-8"))
    return [int(x) for x in data["indices"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-vae", type=int, default=LEAP_VAE_EPOCHS)
    ap.add_argument("--epochs-cfm", type=int, default=LEAP_CFM_EPOCHS)
    ap.add_argument("--cfrl-steps", type=int, default=CFRL_TRAIN_STEPS_NOTEBOOK)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--force-cfproto", action="store_true")
    ap.add_argument("--force-cfrl", action="store_true")
    ap.add_argument("--force-leap", action="store_true")
    ap.add_argument("--skip-cfproto", action="store_true")
    ap.add_argument("--skip-cfrl", action="store_true")
    ap.add_argument("--skip-leap", action="store_true")
    ap.add_argument("--mnist-root", type=Path, default=_MNIST_DIR / "data")
    ap.add_argument(
        "--train-pool",
        type=Path,
        default=_MNIST_DIR / "data_subset" / "train_pool_index.json",
    )
    ap.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Cap CFProto/CFRL train pool size. 0 = use the full pool.",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = _MNIST_DIR / "models" / CKPT_MNIST_CNN
    models_dir = _MNIST_DIR / "models"
    vae_path = _MNIST_DIR / "models" / CKPT_LEAP_VAE
    cfm_path = _MNIST_DIR / "models" / CKPT_LEAP_CFM
    if not weights.is_file():
        raise SystemExit(f"Missing {weights}; run train_classifier.py first.")

    (x_train, y_train, _ohe), (_x_te, _y_te, _ohe_te) = load_mnist_alibi(str(args.mnist_root))
    x_nchw = torch.tensor(nhwc_to_nchw(x_train), dtype=torch.float32, device=device)

    clf = MNISTNet().to(device)
    clf.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)

    models_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_cfproto or not args.skip_cfrl:
        if not args.train_pool.is_file():
            raise SystemExit(f"Missing {args.train_pool}; run select_eval_samples.py first.")
        pool = _load_pool_indices(args.train_pool)
        if args.max_train_images > 0:
            pool = pool[: args.max_train_images]
        x_fit = np.asarray(x_train[pool], dtype=np.float32)
        y_fit = np.asarray(y_train[pool], dtype=np.int64)
        print(f"CFProto/CFRL train pool: {len(x_fit)} images")

        cfproto_ckpt = models_dir / CKPT_CFPROTO
        force_cfproto = args.force or args.force_cfproto
        if not args.skip_cfproto and (force_cfproto or not cfproto_ckpt.is_file()):
            proto = CounterfactualProtoPT(
                clf,
                device=device,
                kappa=0.0,
                beta=0.1,
                gamma=CFPROTO_GAMMA,
                theta=CFPROTO_THETA,
                learning_rate=CFPROTO_LR,
                max_iterations=CF_MAX_ITER,
                c_init=CFPROTO_C_INIT,
                c_steps=CFPROTO_C_STEPS,
                feature_range=CFPROTO_FEATURE_RANGE,
                ae_train_epochs=AE_EPOCHS_CFPROTO,
                ae_batch_size=AE_BATCH_KERAS,
            )
            proto.fit(x_fit, y_fit)
            proto.save_checkpoint(cfproto_ckpt)
            print(f"Saved CFProto checkpoint {cfproto_ckpt}")
        elif not args.skip_cfproto:
            print(f"Skip CFProto ({cfproto_ckpt} exists).")

        cfrl_dir = models_dir / CKPT_CFRL_DIR
        force_cfrl = args.force or args.force_cfrl
        cfrl_ready = cfrl_checkpoint_dir_ok(cfrl_dir) and cfrl_manifest_matches(
            cfrl_dir, args.cfrl_steps
        )
        if not args.skip_cfrl and (force_cfrl or not cfrl_ready):
            explainer = train_cfrl_pytorch_explainer(x_fit, args.cfrl_steps, device, clf)
            cfrl_dir.mkdir(parents=True, exist_ok=True)
            explainer.save(str(cfrl_dir))
            write_cfrl_manifest(cfrl_dir, args.cfrl_steps)
            print(f"Saved CFRL checkpoint dir {cfrl_dir}")
        elif not args.skip_cfrl:
            print(f"Skip CFRL ({cfrl_dir} exists).")

    if args.skip_leap:
        return

    force_leap = args.force or args.force_leap
    if not force_leap and vae_path.is_file() and cfm_path.is_file():
        print("Skip Leap training (checkpoints exist).")
        return

    latent = LEAP_LATENT_DIM
    vae = MNISTVAELatent32(latent_dim=latent).to(device)
    if vae_path.is_file() and not force_leap:
        vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
        print(f"Loaded VAE skip train: {vae_path}")
    else:
        opt_v = torch.optim.Adam(vae.parameters(), lr=LEAP_VAE_LR)
        vae.train()
        ds = TensorDataset(x_nchw)
        dl = DataLoader(ds, batch_size=LEAP_VAE_BATCH, shuffle=True, drop_last=False)
        for _ in tqdm(range(args.epochs_vae), desc="VAE"):
            for (xb,) in dl:
                opt_v.zero_grad(set_to_none=True)
                recon, mu, logvar = vae(xb)
                rec_loss = F.mse_loss(recon, xb)
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = rec_loss + LEAP_VAE_KL_W * kl
                loss.backward()
                opt_v.step()
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vae.state_dict(), vae_path)
        print(f"Saved VAE: {vae_path}")

    vae.eval()
    cfm = MLP_CFM(
        dim=latent,
        time_varying=True,
        cond=True,
        cond_dim=10,
        w=LEAP_MLP_HIDDEN,
    ).to(device)
    if cfm_path.is_file() and not force_leap:
        cfm.load_state_dict(torch.load(cfm_path, map_location=device, weights_only=True))
        print(f"Loaded CFM skip train: {cfm_path}")
    else:
        from torchcfm import ConditionalFlowMatcher

        fm = ConditionalFlowMatcher(sigma=LEAP_CFM_SIGMA)
        opt_c = torch.optim.Adam(cfm.parameters(), lr=LEAP_CFM_LR)
        n = len(x_nchw)
        cfm.train()
        for _ in tqdm(range(args.epochs_cfm), desc="CFM"):
            perm = torch.randperm(n, device=device)
            for s in range(0, n, LEAP_CFM_BATCH):
                idx = perm[s : s + LEAP_CFM_BATCH]
                xb = x_nchw[idx]
                with torch.no_grad():
                    mu, _ = vae.encode(xb)
                    x1 = mu
                    y = clf.predict_proba(xb)
                x0 = torch.randn_like(x1)
                t, xt, ut = fm.sample_location_and_conditional_flow(x0, x1)
                vt = cfm(t, torch.cat([xt, y], dim=-1))
                loss = torch.mean((vt - ut) ** 2)
                opt_c.zero_grad(set_to_none=True)
                loss.backward()
                opt_c.step()
        cfm_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cfm.state_dict(), cfm_path)
        print(f"Saved CFM: {cfm_path}")


if __name__ == "__main__":
    main()
