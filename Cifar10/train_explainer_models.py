"""Train CFProto bundle, CFRL, and Leap VAE+CFM on train pool (disjoint from test eval)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_CIFAR_DIR = Path(__file__).resolve().parent
_GRACO_ROOT = _CIFAR_DIR.parent
sys.path.insert(0, str(_GRACO_ROOT))
sys.path.insert(0, str(_CIFAR_DIR))

from cfrl_runner_cifar import train_cfrl_cifar_explainer, write_cfrl_manifest
from cifar_alibi_config import (
    AE_BATCH_CFPROTO,
    AE_EPOCHS_CFPROTO,
    CKPT_CFPROTO,
    CKPT_CFRL_DIR,
    CKPT_CIFAR_RESNET,
    CKPT_LEAP_CFM,
    CKPT_LEAP_VAE,
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
from cifar_alibi_data import load_cifar10_alibi, nhwc_to_nchw
from common.leap_wae import build_leap_wae_adapter, train_wae_mmd
from explainers.cifar_resnet import CifarResNet
from explainers.cfproto_pt import CounterfactualProtoPT
from explainers.cfm_mlp import MLP_CFM


def _load_pool_indices(train_pool_json: Path) -> list[int]:
    data = json.loads(train_pool_json.read_text(encoding="utf-8"))
    return [int(x) for x in data["indices"]]


def _cfm_conditioning(
    clf: CifarResNet,
    xb: torch.Tensor,
    yb: torch.Tensor,
    mode: str,
) -> torch.Tensor:
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
    ap.add_argument("--cifar-root", type=Path, default=_CIFAR_DIR / "data")
    ap.add_argument(
        "--train-pool",
        type=Path,
        default=_CIFAR_DIR / "data_subset" / "train_pool_index.json",
    )
    ap.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Cap train pool size (first N indices). 0 = use full pool (~50k).",
    )
    ap.add_argument("--cfrl-steps", type=int, default=CFRL_TRAIN_STEPS_NOTEBOOK)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-cfrl", action="store_true")
    ap.add_argument("--skip-leap", action="store_true")
    ap.add_argument("--skip-cfproto", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = _CIFAR_DIR / "models" / CKPT_CIFAR_RESNET
    if not weights.is_file():
        raise SystemExit(f"Missing {weights}; run train_classifier.py first.")

    clf = CifarResNet().to(device)
    clf.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    clf.eval()
    for p in clf.parameters():
        p.requires_grad_(False)

    pool = _load_pool_indices(args.train_pool)
    (x_train, y_train, _ohe), (_x_te, _y_te, _ohe_te) = load_cifar10_alibi(str(args.cifar_root))
    if args.max_train_images is not None and args.max_train_images > 0:
        pool = pool[: args.max_train_images]

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for i in pool:
        xs.append(np.asarray(x_train[int(i)], dtype=np.float32))
        ys.append(int(y_train[int(i)]))
    x_np = np.stack(xs, axis=0)
    y_np = np.asarray(ys, dtype=np.int64)
    print(f"Train pool: {len(x_np)} images")

    models_dir = _CIFAR_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cfproto_ckpt = models_dir / CKPT_CFPROTO
    if not args.skip_cfproto and (args.force or not cfproto_ckpt.is_file()):
        proto = CounterfactualProtoPT(
            clf,
            device=device,
            num_classes=NUM_CLASSES,
            ae_train_epochs=AE_EPOCHS_CFPROTO,
            ae_batch_size=AE_BATCH_CFPROTO,
        )
        proto.fit(x_np, y_np)
        proto.save_checkpoint(cfproto_ckpt)
        print(f"Saved CFProto checkpoint {cfproto_ckpt}")
    elif not args.skip_cfproto:
        print(f"Skip CFProto ({cfproto_ckpt} exists).")

    cfrl_dir = models_dir / CKPT_CFRL_DIR
    if not args.skip_cfrl and (args.force or not cfrl_dir.is_dir()):
        explainer = train_cfrl_cifar_explainer(x_np, args.cfrl_steps, device, clf)
        cfrl_dir.mkdir(parents=True, exist_ok=True)
        explainer.save(str(cfrl_dir))
        write_cfrl_manifest(cfrl_dir, args.cfrl_steps)
        print(f"Saved CFRL dir {cfrl_dir}")
    elif not args.skip_cfrl:
        print(f"Skip CFRL ({cfrl_dir} exists).")

    x_nchw = torch.tensor(nhwc_to_nchw(x_np), dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.long, device=device)
    vae_path = models_dir / CKPT_LEAP_VAE
    cfm_path = models_dir / CKPT_LEAP_CFM
    need_vae = not args.skip_leap and (args.force or not vae_path.is_file())
    need_cfm = not args.skip_leap and (args.force or not cfm_path.is_file() or need_vae)
    if need_vae or need_cfm:
        latent = LEAP_LATENT_DIM
        if need_vae:
            print("Training Leap WAE_MMD (CIFAR-10)...")
            train_wae_mmd(
                x_nchw,
                dataset="cifar10",
                save_path=vae_path,
                device=device,
                batch_size=LEAP_VAE_BATCH,
                train_params={"max_epochs": LEAP_VAE_EPOCHS},
                use_amp=LEAP_WAE_AMP,
            )
        vae = build_leap_wae_adapter("cifar10", device=device)
        vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
        vae.eval()

        if not need_cfm:
            print("Skip Leap CFM (checkpoint exists).")
        else:
            cfm = MLP_CFM(
                dim=latent,
                time_varying=True,
                cond=True,
                cond_dim=NUM_CLASSES,
                w=LEAP_MLP_HIDDEN,
                n_hidden_layers=LEAP_CFM_LAYERS,
            ).to(device)
            from torchcfm import ConditionalFlowMatcher

            fm = ConditionalFlowMatcher(sigma=LEAP_CFM_SIGMA)
            opt_c = torch.optim.Adam(cfm.parameters(), lr=LEAP_CFM_LR)
            n = len(x_nchw)
            cfm.train()
            for _ in tqdm(range(LEAP_CFM_EPOCHS), desc="Leap CFM"):
                perm = torch.randperm(n, device=device)
                for s in range(0, n, LEAP_CFM_BATCH):
                    idx = perm[s : s + LEAP_CFM_BATCH]
                    xb = x_nchw[idx]
                    yb = y_t[idx]
                    with torch.no_grad():
                        mu_z, _ = vae.encode(xb)
                        x1 = mu_z
                        y = _cfm_conditioning(clf, xb, yb, LEAP_CFM_COND_MODE)
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
        print("Skip Leap (checkpoints exist).")


if __name__ == "__main__":
    main()
