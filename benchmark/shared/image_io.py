"""Tensor layout / value-range conversions between benchmark loaders and Graco turtle."""

from __future__ import annotations

import numpy as np
import torch


def tensor_to_numpy_hw(img: torch.Tensor) -> np.ndarray:
    """CHW tensor -> HxW or HxWxC numpy for turtle editing."""
    x = img.detach().cpu().float()
    if x.ndim == 3 and x.shape[0] in (1, 3):
        if x.shape[0] == 1:
            return x[0].numpy()
        return x.permute(1, 2, 0).numpy()
    if x.ndim == 2:
        return x.numpy()
    raise ValueError(f"Expected CHW or HW image, got {tuple(x.shape)}")


def numpy_hw_to_tensor(img: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    """HxW or HxWxC numpy -> CHW tensor matching `like` dtype/device."""
    if img.ndim == 2:
        t = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
    else:
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
    return t.to(device=like.device, dtype=like.dtype)


def morphomnist_to_turtle(img: torch.Tensor) -> np.ndarray:
    """Benchmark MorphoMNIST image in [-1,1] -> turtle canvas same range, HxW."""
    return tensor_to_numpy_hw(img)


def celeba_to_turtle(img: torch.Tensor) -> np.ndarray:
    """Benchmark CelebA [0,1] CHW -> HxWx3 numpy in [0,1]."""
    return tensor_to_numpy_hw(img)


def turtle_to_morphomnist(img: np.ndarray, template: torch.Tensor) -> torch.Tensor:
    return numpy_hw_to_tensor(img, template)


def turtle_to_celeba(img: np.ndarray, template: torch.Tensor) -> torch.Tensor:
    out = np.clip(img, 0.0, 1.0) if img.ndim == 3 else np.clip(img, 0.0, 1.0)
    return numpy_hw_to_tensor(out, template)
