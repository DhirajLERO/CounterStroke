"""CLIP ViT-L/14@336 helpers — thin facade over ``clip_cmmd``."""

from __future__ import annotations

from .clip_cmmd import build_clip_vit_l_14_336, embed_images_rgb01

__all__ = ["build_clip_vit_l_14_336", "embed_images_rgb01"]
