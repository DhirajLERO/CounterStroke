"""CLIP image embeddings in PyTorch only (no google-research / scenic / JAX)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def build_clip_vit_l_14_336(
    device: torch.device,
    pretrained: str = "openai",
) -> tuple[nn.Module, Callable[[Image.Image], torch.Tensor]]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-336",
        pretrained=pretrained,
    )
    model = model.to(device).eval()
    return model, preprocess


def _pil_from_rgb01(arr: np.ndarray) -> Image.Image:
    a = np.asarray(arr, dtype=np.float64)
    a = np.clip(a, 0.0, 1.0)
    u8 = (a * 255.0).round().astype(np.uint8)
    return Image.fromarray(u8, mode="RGB")


def embed_images_rgb01(
    model: nn.Module,
    preprocess: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    images_rgb_01: list[np.ndarray],
    batch_size: int,
) -> np.ndarray:
    out: list[np.ndarray] = []
    n = len(images_rgb_01)
    for s in range(0, n, batch_size):
        chunk = images_rgb_01[s : s + batch_size]
        tensors = [preprocess(_pil_from_rgb01(x)) for x in chunk]
        xb = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            feat = model.encode_image(xb).float()
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        out.append(feat.cpu().numpy().astype(np.float64))
    if not out:
        return np.zeros((0, 0), dtype=np.float64)
    return np.concatenate(out, axis=0)
