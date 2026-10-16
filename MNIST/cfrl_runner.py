"""CFRL training + batched explain (Alibi)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alibi_mnist_config import (
    CFRL_AE_BATCH,
    CFRL_AE_EPOCHS,
    CFRL_BATCH_SIZE,
    CFRL_BUFFER_SIZE,
    CFRL_COEFF_CONSISTENCY,
    CFRL_COEFF_SPARSITY,
    CFRL_LATENT_DIM,
    CKPT_CFRL_MANIFEST,
)
from mnist_alibi_data import alibi_nhwc_to_cfrl01_nchw, nchw_to_nhwc, np_mnist_to_nchw


def pred_np_cfrl_factory(pt: torch.nn.Module, device: torch.device):
    def pred_np_cfrl(X: np.ndarray) -> np.ndarray:
        x_nchw = np_mnist_to_nchw(X)
        if float(x_nchw.min()) >= -1e-4:
            x_nchw = x_nchw - 0.5
        xt = torch.tensor(x_nchw, dtype=torch.float32, device=device)
        with torch.no_grad():
            return pt.predict_proba(xt).cpu().numpy()

    return pred_np_cfrl


def train_cfrl_pytorch_explainer(
    x_train_nhwc: np.ndarray,
    train_steps: int,
    device: torch.device,
    pt: torch.nn.Module,
) -> Any:
    from alibi.explainers import CounterfactualRL
    from alibi.models.pytorch.autoencoder import AE as PTAE
    from alibi.models.pytorch.cfrl_models import MNISTDecoder as PTDec
    from alibi.models.pytorch.cfrl_models import MNISTEncoder as PTEnc

    pred_np_cfrl = pred_np_cfrl_factory(pt, device)
    latent = CFRL_LATENT_DIM
    enc_pt = PTEnc(latent_dim=latent)
    dec_pt = PTDec(latent_dim=latent)
    ae_pt = PTAE(encoder=enc_pt, decoder=dec_pt).to(device)
    opt_pt = torch.optim.Adam(ae_pt.parameters(), lr=1e-3)
    x_train_01 = alibi_nhwc_to_cfrl01_nchw(x_train_nhwc)
    x_t = torch.tensor(x_train_01, dtype=torch.float32, device=device)
    ae_pt.train()
    for _ in range(CFRL_AE_EPOCHS):
        perm = torch.randperm(len(x_t))
        for s in range(0, len(x_t), CFRL_AE_BATCH):
            idx = perm[s : s + CFRL_AE_BATCH]
            b = x_t[idx]
            opt_pt.zero_grad()
            rec01 = ae_pt(b)
            loss = torch.nn.functional.mse_loss(rec01, b)
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
    y_t = np.zeros((k, 10), dtype=np.float32)
    y_t[np.arange(k), target_classes.astype(np.int64)] = 1.0
    x_01 = alibi_nhwc_to_cfrl01_nchw(x_nhwc_batch)
    t0 = time.perf_counter()
    exp = explainer.explain(x_01, Y_t=y_t, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    x_cf = np.asarray(exp.data["cf"]["X"], dtype=np.float32) - 0.5
    if x_cf.ndim == 4 and x_cf.shape[1] == 1:
        cf_nhwc = nchw_to_nhwc(x_cf)
    else:
        cf_nhwc = x_cf
    y_cf = exp.data["cf"]["class"].flatten().astype(np.int64)
    hits = y_cf == target_classes.astype(np.int64)
    return cf_nhwc, hits, elapsed
