"""CFRL training + batched explain for CIFAR-10 RGB (32×32)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cifar_alibi_config import (
    CFRL_AE_BATCH,
    CFRL_AE_EPOCHS,
    CFRL_BATCH_SIZE,
    CFRL_BUFFER_SIZE,
    CFRL_COEFF_CONSISTENCY,
    CFRL_COEFF_SPARSITY,
    CFRL_LATENT_DIM,
    CKPT_CFRL_MANIFEST,
    NUM_CLASSES,
)
from explainers.cfrl_rgb_models_cifar import CFRGBDecoderCifar, CFRGBEncoderCifar


def _numpy_batch_to_alibi_nchw_pt(X: np.ndarray) -> np.ndarray:
    """Alibi CFRL may pass NCHW [0,1] or NHWC Alibi — normalize to NCHW Alibi [-0.5, 0.5]."""
    x = np.asarray(X, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected ndim=4, got {x.shape}")
    if x.shape[1] == 3:
        x01 = np.clip(x, 0.0, 1.0)
        return x01 - 0.5
    if x.shape[-1] == 3:
        return np.transpose(x, (0, 3, 1, 2))
    raise ValueError(f"Unrecognized RGB layout: {x.shape}")


def _nhwc_rgb_alibi_to_nchw_01(x: np.ndarray) -> np.ndarray:
    t = np.transpose(np.asarray(x, dtype=np.float32), (0, 3, 1, 2))
    return np.clip(t + 0.5, 0.0, 1.0)


def _nchw_01_to_nhwc_alibi(x: np.ndarray) -> np.ndarray:
    t = np.asarray(x, dtype=np.float32)
    return np.transpose(t - 0.5, (0, 2, 3, 1))


def pred_np_cfrl_factory(pt: torch.nn.Module, device: torch.device):
    def pred_np_cfrl(X: np.ndarray) -> np.ndarray:
        x_alibi = _numpy_batch_to_alibi_nchw_pt(X)
        xt = torch.tensor(x_alibi, dtype=torch.float32, device=device)
        with torch.no_grad():
            return pt.predict_proba(xt).cpu().numpy()

    return pred_np_cfrl


def train_cfrl_cifar_explainer(
    x_train_nhwc: np.ndarray,
    train_steps: int,
    device: torch.device,
    pt: torch.nn.Module,
) -> Any:
    from alibi.explainers import CounterfactualRL
    from alibi.models.pytorch.autoencoder import AE as PTAE

    pred_np_cfrl = pred_np_cfrl_factory(pt, device)
    latent = CFRL_LATENT_DIM
    enc_pt = CFRGBEncoderCifar(latent_dim=latent).to(device)
    dec_pt = CFRGBDecoderCifar(latent_dim=latent).to(device)
    ae_pt = PTAE(encoder=enc_pt, decoder=dec_pt).to(device)
    opt_pt = torch.optim.Adam(ae_pt.parameters(), lr=1e-3)
    x_train_01 = _nhwc_rgb_alibi_to_nchw_01(x_train_nhwc)
    x_t = torch.tensor(x_train_01, dtype=torch.float32, device=device)
    ae_pt.train()
    for _ in range(CFRL_AE_EPOCHS):
        perm = torch.randperm(len(x_t))
        for s in range(0, len(x_t), CFRL_AE_BATCH):
            idx = perm[s : s + CFRL_AE_BATCH]
            b = x_t[idx]
            opt_pt.zero_grad()
            rec01 = ae_pt(b)
            loss = F.mse_loss(rec01, b)
            loss.backward()
            opt_pt.step()
    ae_pt.eval()

    cfrl = CounterfactualRL(
        predictor=pred_np_cfrl,
        encoder=enc_pt,
        decoder=dec_pt,
        coeff_sparsity=CFRL_COEFF_SPARSITY,
        coeff_consistency=CFRL_COEFF_CONSISTENCY,
        latent_dim=latent,
        backend="pytorch",
        train_steps=train_steps,
        batch_size=CFRL_BATCH_SIZE,
        replay_buffer_size=CFRL_BUFFER_SIZE,
        exploration_steps=min(50, train_steps // 2),
        update_after=5,
    )
    cfrl.fit(x_train_01)
    return cfrl


def cfrl_checkpoint_dir_ok(d: Path) -> bool:
    if not d.is_dir():
        return False
    req = (
        "meta.dill",
        "explainer.dill",
        "encoder.pth",
        "decoder.pth",
        "actor.pth",
        "critic.pth",
    )
    return all((d / f).is_file() for f in req)


def cfrl_manifest_matches(d: Path, train_steps: int) -> bool:
    m = d / CKPT_CFRL_MANIFEST
    if not m.is_file():
        return True
    try:
        data = json.loads(m.read_text())
    except json.JSONDecodeError:
        return False
    return int(data.get("train_steps", -1)) == int(train_steps) and int(
        data.get("latent_dim", -1)
    ) == int(CFRL_LATENT_DIM)


def write_cfrl_manifest(d: Path, train_steps: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / CKPT_CFRL_MANIFEST).write_text(
        json.dumps({"train_steps": int(train_steps), "latent_dim": int(CFRL_LATENT_DIM)}),
        encoding="utf-8",
    )


def cfrl_explain_batch_nhwc(
    explainer: Any,
    x_nhwc_batch: np.ndarray,
    target_classes: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    k = int(target_classes.shape[0])
    y_t = np.zeros((k, NUM_CLASSES), dtype=np.float32)
    y_t[np.arange(k), target_classes.astype(np.int64)] = 1.0
    x_01 = _nhwc_rgb_alibi_to_nchw_01(x_nhwc_batch)
    t0 = time.perf_counter()
    exp = explainer.explain(x_01, Y_t=y_t, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    x_cf = np.asarray(exp.data["cf"]["X"], dtype=np.float32)
    cf_nhwc = _nchw_01_to_nhwc_alibi(x_cf)
    y_cf = exp.data["cf"]["class"].flatten().astype(np.int64)
    hits = y_cf == target_classes.astype(np.int64)
    return cf_nhwc, hits, elapsed
