from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def predict_probs_numpy(
    model: nn.Module,
    imgs_np: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 1024,
    amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched inference on (N,H,W,3) images — Alibi float in [-0.5, 0.5]."""
    if imgs_np.ndim != 4 or imgs_np.shape[-1] != 3:
        raise ValueError(f"imgs_np must be (N,H,W,3), got {imgs_np.shape}")

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    x = torch.from_numpy(imgs_np).float().permute(0, 3, 1, 2)

    loader = DataLoader(TensorDataset(x), batch_size=int(batch_size), shuffle=False)

    model.eval()
    pred_cls_list = []
    pred_prob_list = []
    probs_list = []

    use_amp = bool(amp) and dev.type == "cuda"
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(dev)
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    log_probs = model(xb)
            else:
                log_probs = model(xb)

            probs = torch.exp(log_probs)
            pred_prob, pred_cls = probs.max(dim=1)

            pred_cls_list.append(pred_cls.detach().cpu())
            pred_prob_list.append(pred_prob.detach().cpu())
            probs_list.append(probs.detach().cpu())

    pred_cls_arr = torch.cat(pred_cls_list, dim=0).numpy()
    pred_prob_arr = torch.cat(pred_prob_list, dim=0).numpy()
    probs_arr = torch.cat(probs_list, dim=0).numpy()
    return pred_cls_arr, pred_prob_arr, probs_arr
