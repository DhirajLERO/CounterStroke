"""Attribute-intervention fitness for Graco (minimize predictor error vs targets)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _broadcast_targets(
    targets: dict[str, torch.Tensor],
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Expand single-sample intervention targets to match candidate batch size."""
    out: dict[str, torch.Tensor] = {}
    for key, value in targets.items():
        if key == "image":
            continue
        t = value.to(device)
        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t.view(1, -1)
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size, *t.shape[1:])
        elif t.shape[0] != batch_size:
            t = t[:1].expand(batch_size, *t.shape[1:])
        out[key] = t
    return out


def intervention_loss_morphomnist(
    predictors: dict,
    images_nchw: torch.Tensor,
    targets: dict[str, torch.Tensor],
    device: str = "cuda",
) -> np.ndarray:
    """Per-sample scalar loss (lower = better effectiveness)."""
    x = images_nchw.to(device)
    batch_size = x.shape[0]
    batch_t = _broadcast_targets(targets, batch_size, device)
    losses = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        for key, clf in predictors.items():
            if key not in batch_t:
                continue
            tgt = batch_t[key]
            if key == "thickness":
                cond = batch_t.get("intensity")
                if cond is None:
                    continue
                pred = clf(x, cond)
                losses += F.mse_loss(pred, tgt.float().view_as(pred), reduction="none").view(batch_size, -1).mean(
                    dim=1
                )
            elif key == "digit":
                pred = clf(x)
                losses += F.cross_entropy(pred, tgt.argmax(-1), reduction="none")
            else:
                pred = torch.sigmoid(clf(x))
                losses += F.mse_loss(pred, tgt.float().view_as(pred), reduction="none").view(batch_size, -1).mean(
                    dim=1
                )
    return losses.cpu().numpy().astype(np.float32)


def intervention_loss_celeba(
    predictors: dict,
    images_nchw: torch.Tensor,
    targets: dict[str, torch.Tensor],
    device: str = "cuda",
) -> np.ndarray:
    x = images_nchw.to(device)
    batch_size = x.shape[0]
    batch_t = _broadcast_targets(targets, batch_size, device)
    losses = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        for key, clf in predictors.items():
            if key not in batch_t:
                continue
            tgt = batch_t[key]
            if getattr(clf, "conditions", None) and clf.conditions.get(key):
                cond = torch.cat([batch_t[att] for att in clf.conditions[key]], dim=1)
                pred = torch.sigmoid(clf(x, cond))
            else:
                pred = torch.sigmoid(clf(x))
            losses += F.binary_cross_entropy(pred, tgt.float().view_as(pred), reduction="none").view(
                batch_size, -1
            ).mean(dim=1)
    return losses.cpu().numpy().astype(np.float32)
