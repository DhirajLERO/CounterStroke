"""Adapter: Alibi NCHW [-0.5, 0.5] <-> WAE_MMD [0, 1] Sigmoid outputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import CIFAR10_WAE_MODEL, IMAGENET_PIPELINE_HW, IMAGENET_WAE_MODEL
from .wae_mmd import WAE_MMD


def build_leap_wae_adapter(dataset: str, device: torch.device | None = None) -> "LeapWaeMmdAdapter":
    if dataset == "cifar10":
        cfg = dict(CIFAR10_WAE_MODEL)
        pipeline_hw: tuple[int, int] | None = None
    elif dataset == "imagenet":
        cfg = dict(IMAGENET_WAE_MODEL)
        pipeline_hw = IMAGENET_PIPELINE_HW
    else:
        raise ValueError(f"Unknown dataset {dataset!r}; use cifar10 or imagenet")
    wae = WAE_MMD(**cfg)
    adapter = LeapWaeMmdAdapter(wae, pipeline_hw=pipeline_hw)
    if device is not None:
        adapter = adapter.to(device)
        verify_wae_bottleneck(adapter, device)
    return adapter


def verify_wae_bottleneck(adapter: "LeapWaeMmdAdapter", device: torch.device) -> None:
    """Dry-run encoder flatten size vs ``fc_in`` (catches img_size / stage mismatches)."""
    wae = adapter.wae
    w = int(wae.img_size)
    was_training = adapter.training
    adapter.eval()
    try:
        with torch.no_grad():
            dummy01 = torch.zeros(1, 3, w, w, device=device)
            flat = torch.flatten(wae.encoder(dummy01), start_dim=1)
            if flat.shape[1] != int(wae.embedding_length):
                raise RuntimeError(
                    f"WAE bottleneck mismatch: encoder flat {flat.shape[1]} != "
                    f"embedding_length {wae.embedding_length} (img_size={w})"
                )
            z = wae.encode(dummy01)
            recon = wae.decode(z)
            if recon.shape != dummy01.shape:
                raise RuntimeError(f"WAE decode shape {recon.shape} != encode {dummy01.shape}")
    finally:
        if was_training:
            adapter.train()


class LeapWaeMmdAdapter(nn.Module):
    """
    Drop-in replacement for KL VAE: ``encode`` returns (z, logvar_dummy).

    ImageNet: resizes 299×299 Alibi ↔ 256×256 inside the WAE only (see ``pipeline_hw``).
    """

    def __init__(self, wae: WAE_MMD, *, pipeline_hw: tuple[int, int] | None = None) -> None:
        super().__init__()
        self.wae = wae
        self.latent_dim = int(wae.total_latent_dim)
        self.wae_size = int(wae.img_size)
        self.pipeline_hw = pipeline_hw

    @staticmethod
    def alibi_to_01(x: torch.Tensor) -> torch.Tensor:
        return (x + 0.5).clamp(0.0, 1.0)

    @staticmethod
    def o1_to_alibi(x: torch.Tensor) -> torch.Tensor:
        return x - 0.5

    def _resize_alibi_nchw(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if int(x.shape[-2]) == size[0] and int(x.shape[-1]) == size[1]:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def to_wae_alibi(self, x_alibi: torch.Tensor) -> torch.Tensor:
        """Alibi at pipeline resolution → Alibi at WAE resolution (256 for ImageNet)."""
        if self.pipeline_hw is None:
            return x_alibi
        return self._resize_alibi_nchw(x_alibi, (self.wae_size, self.wae_size))

    def from_wae_alibi(self, x_alibi: torch.Tensor, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
        """Alibi at WAE resolution → Alibi at ``target_hw`` or ``pipeline_hw``."""
        out_hw = target_hw if target_hw is not None else self.pipeline_hw
        if out_hw is None:
            return x_alibi
        return self._resize_alibi_nchw(x_alibi, out_hw)

    def encode(self, x_alibi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_wae = self.to_wae_alibi(x_alibi)
        x01 = self.alibi_to_01(x_wae)
        z = self.wae.encode(x01)
        return z, torch.zeros_like(z)

    def decode(self, z: torch.Tensor, *, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
        recon01 = self.wae.decode(z, cond=None)
        recon_alibi = self.o1_to_alibi(recon01)
        return self.from_wae_alibi(recon_alibi, target_hw=target_hw)

    def forward(self, x_alibi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, lv = self.encode(x_alibi)
        return self.decode(z, target_hw=None if self.pipeline_hw is None else self.pipeline_hw), z, lv
