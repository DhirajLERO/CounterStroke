"""
ImageNet-100 task classifier inference for Graco RGB Alibi tensors (N, 3, 299, 299) in [-0.5, 0.5].
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_task_classifier(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    *,
    num_classes: int = 100,
) -> nn.Module:
    """Load fine-tuned 100-way InceptionV3 from ``train_classifier.py`` checkpoint."""
    import sys

    graco_dir = Path(__file__).resolve().parent
    imnet_dir = graco_dir.parent
    if str(imnet_dir) not in sys.path:
        sys.path.insert(0, str(imnet_dir))
    from explainers.imagenet100_inception import load_imagenet100_classifier

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    return load_imagenet100_classifier(checkpoint, dev, num_classes=num_classes)


def _model_logits(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_logits"):
        return model.forward_logits(xb)
    return model(xb)


def predict_probs_numpy_inception(
    model: nn.Module,
    imgs_np: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
    amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if imgs_np.ndim != 4 or imgs_np.shape[1] != 3:
        raise ValueError(f"imgs_np must be (N,3,H,W), got {imgs_np.shape}")

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    x = torch.from_numpy(np.ascontiguousarray(imgs_np, dtype=np.float32))
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
                    logits = _model_logits(model, xb)
            else:
                logits = _model_logits(model, xb)
            probs = F.softmax(logits, dim=1)
            pred_prob, pred_cls = probs.max(dim=1)

            pred_cls_list.append(pred_cls.detach().cpu())
            pred_prob_list.append(pred_prob.detach().cpu())
            probs_list.append(probs.detach().cpu())

    pred_cls_arr = torch.cat(pred_cls_list, dim=0).numpy().astype(np.int64)
    pred_prob_arr = torch.cat(pred_prob_list, dim=0).numpy().astype(np.float32)
    probs_arr = torch.cat(probs_list, dim=0).numpy().astype(np.float32)
    return pred_cls_arr, pred_prob_arr, probs_arr
