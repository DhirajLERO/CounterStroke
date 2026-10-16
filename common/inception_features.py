"""InceptionV3 ``fc=Identity`` feature extractor for distribution / embedding metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def build_inception_feature_net(device: torch.device) -> nn.Module:
    """Returns Inception v3 trunk producing pooled embeddings (no classifier head)."""
    try:
        inc = models.inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    except TypeError:
        inc = models.inception_v3(pretrained=True, transform_input=False)
    inc.aux_logits = False
    inc.fc = nn.Identity()
    inc.eval().to(device)
    return inc


def rgb_nhwc_alibi_to_inception_batch(x_nhwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N,299,299,3) in [-0.5,0.5] → ImageNet-normalized NCHW for Inception."""
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(x_nhwc, dtype=np.float32)).to(device)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    t = t.permute(0, 3, 1, 2)
    t01 = (t + 0.5).clamp(0.0, 1.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=t01.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=t01.dtype).view(1, 3, 1, 1)
    return (t01 - mean) / std


@torch.no_grad()
def inception_features_from_rgb_nhwc(
    inc: nn.Module,
    x_nhwc: np.ndarray,
    device: torch.device,
    batch: int = 64,
) -> np.ndarray:
    x = np.asarray(x_nhwc, dtype=np.float32)
    if x.ndim == 3:
        x = x[np.newaxis, ...]
    outs: list[np.ndarray] = []
    n = x.shape[0]
    for s in range(0, n, batch):
        chunk = x[s : s + batch]
        inp = rgb_nhwc_alibi_to_inception_batch(chunk, device)
        feat = inc(inp)
        outs.append(feat.float().cpu().numpy())
    return np.concatenate(outs, axis=0)
