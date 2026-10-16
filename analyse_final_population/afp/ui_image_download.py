"""Downloadable raster images with Native / 2× / 4× PNG options."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image

_SCALE_OPTIONS = ("Native", "2×", "4×")
_SCALE_FACTOR = {"Native": 1, "2×": 2, "4×": 4}
_SCALE_TAG = {"Native": "native", "2×": "2x", "4×": "4x"}


def _to_rgb_uint8(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        g = arr[:, :, 0]
        arr = np.stack([g, g, g], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxW or HxWx3 image, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def encode_png(rgb_uint8: np.ndarray, scale: int = 1) -> bytes:
    """Encode an HxWx3 uint8 array as PNG; nearest-neighbor upscale for scale in {1,2,4}."""
    if scale not in (1, 2, 4):
        raise ValueError(f"scale must be 1, 2, or 4; got {scale}")
    rgb = _to_rgb_uint8(rgb_uint8)
    im = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        w, h = im.size
        im = im.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def show_downloadable_image(
    image: Any,
    *,
    key: str,
    filename_stem: str,
    caption: str | None = None,
    width: Any = None,
) -> None:
    """Show an image with a Download popover (Native / 2× / 4× PNG)."""
    kwargs: dict[str, Any] = {}
    if caption is not None:
        kwargs["caption"] = caption
    if width is not None:
        kwargs["width"] = width
    st.image(image, **kwargs)

    rgb = _to_rgb_uint8(image)
    with st.popover("Download"):
        choice = st.radio(
            "Resolution",
            _SCALE_OPTIONS,
            index=0,
            horizontal=True,
            key=f"{key}_res",
        )
        scale = _SCALE_FACTOR[choice]
        tag = _SCALE_TAG[choice]
        st.download_button(
            label=f"Download PNG ({choice})",
            data=encode_png(rgb, scale=scale),
            file_name=f"{filename_stem}_{tag}.png",
            mime="image/png",
            key=f"{key}_dl",
        )
