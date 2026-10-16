"""Plain-PyTorch WAE_MMD training (ported from LeapFactual ExpGAN)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from .configs import GALAXY_TRAIN_PARAMS, IMAGENET_WAE_IMG_SIZE
from .train_utils import frange_cycle_linear
from .wae_adapter import LeapWaeMmdAdapter, build_leap_wae_adapter


def _split_train_val(x: torch.Tensor, val_frac: float = 0.1, seed: int = 0) -> tuple[TensorDataset, TensorDataset]:
    n = len(x)
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(TensorDataset(x), [n_train, n_val], generator=gen)
    return train_ds, val_ds  # type: ignore[return-value]


def train_wae_mmd(
    x_nchw_alibi: torch.Tensor,
    *,
    dataset: str,
    save_path: Path,
    device: torch.device,
    batch_size: int,
    train_params: dict | None = None,
    use_amp: bool = False,
) -> LeapWaeMmdAdapter:
    """
    Train WAE_MMD on Alibi NCHW tensors, save ``LeapWaeMmdAdapter`` state dict to ``save_path``.
    """
    params = dict(GALAXY_TRAIN_PARAMS)
    if train_params:
        params.update(train_params)

    if dataset == "imagenet":
        # Train WAE at 256×256; eval/generation still uses 299×299 via adapter resize
        x_nchw_alibi = F.interpolate(
            x_nchw_alibi,
            size=(IMAGENET_WAE_IMG_SIZE, IMAGENET_WAE_IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        print(f"ImageNet WAE training: resized inputs to {IMAGENET_WAE_IMG_SIZE}×{IMAGENET_WAE_IMG_SIZE}")

    adapter = build_leap_wae_adapter(dataset, device=device)
    wae = adapter.wae
    max_epochs = int(params["max_epochs"])
    val_every = int(params.get("val_every", 10))

    disc_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, p in wae.named_parameters():
        if "discriminator" in name:
            disc_params.append(p)
        else:
            other_params.append(p)

    opt_g = torch.optim.AdamW(other_params, lr=float(params["LR"]), weight_decay=float(params["weight_decay"]))
    opt_d = torch.optim.AdamW(disc_params, lr=float(params["LR_2"]))
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=float(params["scheduler_gamma"]))
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=float(params["scheduler_gamma_2"]))

    gan_sched = frange_cycle_linear(0.0, float(params["gan_weight"]), max_epochs, n_cycle=1, ratio=0.3)
    adv_sched = frange_cycle_linear(0.0, float(params["adv_weight"]), max_epochs, n_cycle=1, ratio=0.1)

    train_ds, val_ds = _split_train_val(x_nchw_alibi)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(x_nchw_alibi) > batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    try:
        from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except Exception:
        ssim_metric = None

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    best_ssim = -1.0
    dummy_labels = torch.zeros(1, dtype=torch.long)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        wae.train()
        m_g = float(gan_sched[epoch])
        m_a = float(adv_sched[epoch])

        for (xb_alibi,) in tqdm(train_loader, desc=f"WAE_MMD e{epoch+1}/{max_epochs}", leave=False):
            xb = adapter.alibi_to_01(adapter.to_wae_alibi(xb_alibi.to(device)))
            labels = dummy_labels.expand(xb.shape[0]).to(device)

            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                results = wae(xb, labels=labels)
                loss_g = wae.loss_function(
                    *results,
                    real_img=xb,
                    labels=labels,
                    M_P=float(params["perceptual_weight"]),
                    M_R=float(params["recon_weight"]),
                    M_G=m_g,
                    M_A=m_a,
                )["loss"]
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(other_params, float(params["gradient_clip_val"]))
            scaler.step(opt_g)
            scaler.update()

            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                recon_pred, z_pred = wae.forward_gan(xb, labels=labels)
            loss_d = wae.loss_adv(recon_pred, z_pred, real_img=xb, labels=labels)["D_loss"]
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc_params, float(params["gradient_clip_val"]))
            opt_d.step()

        sched_g.step()
        sched_d.step()

        if ssim_metric is not None and ((epoch + 1) % val_every == 0 or epoch == max_epochs - 1):
            wae.eval()
            scores: list[float] = []
            with torch.no_grad():
                for (vb_alibi,) in val_loader:
                    vb = adapter.alibi_to_01(adapter.to_wae_alibi(vb_alibi.to(device)))
                    recon = wae.generate(vb, labels=dummy_labels.expand(vb.shape[0]).to(device))
                    scores.append(float(ssim_metric(recon, vb).item()))
            mean_ssim = sum(scores) / max(len(scores), 1)
            print(f"  epoch {epoch+1} val_ssim={mean_ssim:.4f}")
            if mean_ssim > best_ssim:
                best_ssim = mean_ssim
                torch.save(adapter.state_dict(), save_path)
                print(f"  saved best WAE to {save_path}")

    if best_ssim < 0:
        torch.save(adapter.state_dict(), save_path)
        print(f"  saved WAE to {save_path} (no SSIM metric)")

    adapter.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    adapter.eval()
    return adapter


def train_wae_mmd_dataloader(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    dataset: str,
    save_path: Path,
    device: torch.device,
    train_params: dict | None = None,
    use_amp: bool = False,
) -> LeapWaeMmdAdapter:
    """Train WAE_MMD from streaming DataLoaders (ImageNet-100 scale)."""
    params = dict(GALAXY_TRAIN_PARAMS)
    if train_params:
        params.update(train_params)

    adapter = build_leap_wae_adapter(dataset, device=device)
    wae = adapter.wae
    max_epochs = int(params["max_epochs"])
    val_every = int(params.get("val_every", 10))

    disc_params: list[nn.Parameter] = []
    other_params: list[nn.Parameter] = []
    for name, p in wae.named_parameters():
        if "discriminator" in name:
            disc_params.append(p)
        else:
            other_params.append(p)

    opt_g = torch.optim.AdamW(other_params, lr=float(params["LR"]), weight_decay=float(params["weight_decay"]))
    opt_d = torch.optim.AdamW(disc_params, lr=float(params["LR_2"]))
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=float(params["scheduler_gamma"]))
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=float(params["scheduler_gamma_2"]))

    gan_sched = frange_cycle_linear(0.0, float(params["gan_weight"]), max_epochs, n_cycle=1, ratio=0.3)
    adv_sched = frange_cycle_linear(0.0, float(params["adv_weight"]), max_epochs, n_cycle=1, ratio=0.1)

    try:
        from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except Exception:
        ssim_metric = None

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    best_ssim = -1.0
    dummy_labels = torch.zeros(1, dtype=torch.long)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        wae.train()
        m_g = float(gan_sched[epoch])
        m_a = float(adv_sched[epoch])

        for xb_alibi, _yb in tqdm(train_loader, desc=f"WAE_MMD e{epoch+1}/{max_epochs}", leave=False):
            xb_alibi = xb_alibi.to(device, non_blocking=True)
            xb_wae = adapter.to_wae_alibi(xb_alibi)
            xb = adapter.alibi_to_01(xb_wae)
            labels = dummy_labels.expand(xb.shape[0]).to(device)

            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                results = wae(xb, labels=labels)
                loss_g = wae.loss_function(
                    *results,
                    real_img=xb,
                    labels=labels,
                    M_P=float(params["perceptual_weight"]),
                    M_R=float(params["recon_weight"]),
                    M_G=m_g,
                    M_A=m_a,
                )["loss"]
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(other_params, float(params["gradient_clip_val"]))
            scaler.step(opt_g)
            scaler.update()

            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                recon_pred, z_pred = wae.forward_gan(xb, labels=labels)
            loss_d = wae.loss_adv(recon_pred, z_pred, real_img=xb, labels=labels)["D_loss"]
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc_params, float(params["gradient_clip_val"]))
            opt_d.step()

        sched_g.step()
        sched_d.step()

        if ssim_metric is not None and ((epoch + 1) % val_every == 0 or epoch == max_epochs - 1):
            wae.eval()
            scores: list[float] = []
            with torch.no_grad():
                for vb_alibi, _yb in val_loader:
                    vb_alibi = vb_alibi.to(device, non_blocking=True)
                    vb_wae = adapter.to_wae_alibi(vb_alibi)
                    vb = adapter.alibi_to_01(vb_wae)
                    recon = wae.generate(vb, labels=dummy_labels.expand(vb.shape[0]).to(device))
                    scores.append(float(ssim_metric(recon, vb).item()))
            mean_ssim = sum(scores) / max(len(scores), 1)
            print(f"  epoch {epoch+1} val_ssim={mean_ssim:.4f}")
            if mean_ssim > best_ssim:
                best_ssim = mean_ssim
                torch.save(adapter.state_dict(), save_path)
                print(f"  saved best WAE to {save_path}")

    if best_ssim < 0:
        torch.save(adapter.state_dict(), save_path)
        print(f"  saved WAE to {save_path} (no SSIM metric)")

    adapter.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    adapter.eval()
    return adapter
