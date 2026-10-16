"""Matplotlib figures and Streamlit styling."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from .class_names import label_name
from .config_ui import COLOR_INK, COLOR_LAVENDER, COLOR_MINT, COLOR_PEACH
from .inputs import alibi_to_display_01, display_to_rgb_uint8

_CHART_FACE = "#faf8ff"
_CHART_AX = "#ffffff"


def inject_custom_css() -> str:
    return """
<style>
    .stApp {
        background: linear-gradient(145deg, #f5f0ff 0%, #e8f4f8 45%, #fff5eb 100%);
    }
    h1, h2, h3, h4, .stMarkdown { color: #3d3a5c !important; }
    div[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.85);
        border-right: 1px solid #e0dce8;
    }
    .paper-card, .dataset-card {
        background: rgba(255,255,255,0.92);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 24px rgba(61,58,92,0.08);
        border: 1px solid #ebe6f5;
    }
    .dataset-card { min-height: 140px; }
    .abstract { color: #6b6888; line-height: 1.55; }
    .toolbar-row { gap: 0.5rem; }
    .sample-selected {
        outline: 3px solid #9ee7c4;
        border-radius: 8px;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.7);
        border-radius: 12px;
        padding: 8px;
    }
</style>
"""


def _chart_style(ax) -> None:
    ax.set_facecolor(_CHART_AX)
    ax.tick_params(colors=COLOR_INK)
    for spine in ax.spines.values():
        spine.set_color("#ddd8ea")


def fig_class_distribution(
    dist: dict[int, int],
    class_names: list[str],
    *,
    title: str,
) -> plt.Figure:
    ids = sorted(dist.keys())
    counts = [dist[i] for i in ids]
    labels = [label_name(class_names, i) for i in ids]
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=_CHART_FACE)
    colors = [COLOR_LAVENDER, COLOR_MINT, COLOR_PEACH] * (len(ids) // 3 + 1)
    ax.bar(range(len(ids)), counts, color=colors[: len(ids)], edgecolor="#ddd8ea")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color=COLOR_INK)
    ax.set_ylabel("Count", color=COLOR_INK)
    ax.set_title(title, color=COLOR_INK, fontsize=11)
    _chart_style(ax)
    fig.tight_layout()
    return fig


def fig_softmax_bars(
    probs: np.ndarray,
    class_names: list[str],
    *,
    title: str,
    highlight_idx: int | None = None,
    max_bars: int = 100,
) -> plt.Figure:
    n = len(probs)
    if n > max_bars:
        top_idx = np.argsort(probs)[-max_bars:][::-1]
    else:
        top_idx = np.arange(n)
    labels = [label_name(class_names, int(i)) for i in top_idx]
    vals = [float(probs[i]) for i in top_idx]
    colors = [COLOR_MINT if int(i) == highlight_idx else COLOR_LAVENDER for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, max(3, len(top_idx) * 0.22)), facecolor=_CHART_FACE)
    y_pos = np.arange(len(top_idx))
    ax.barh(y_pos, vals, color=colors, edgecolor="#ddd8ea")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7, color=COLOR_INK)
    ax.set_xlabel("Probability", color=COLOR_INK)
    ax.set_title(title, color=COLOR_INK, fontsize=11)
    ax.invert_yaxis()
    _chart_style(ax)
    fig.tight_layout()
    return fig


def fig_heatmap_landscape(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
) -> plt.Figure:
    fig, ax = plt.subplots(
        figsize=(max(6, len(col_labels) * 1.2), max(4, len(row_labels) * 0.28)),
        facecolor=_CHART_FACE,
    )
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", color=COLOR_INK, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7, color=COLOR_INK)
    ax.set_title(title, color=COLOR_INK, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    return fig


def fig_o1_histogram(o1_values: list[float], *, title: str) -> plt.Figure | None:
    if not o1_values:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=_CHART_FACE)
    ax.hist(o1_values, bins=30, color=COLOR_PEACH, edgecolor="#e8c4a8", alpha=0.9)
    ax.set_xlabel("O1 (p_blank − p_target)", color=COLOR_INK)
    ax.set_ylabel("Count", color=COLOR_INK)
    ax.set_title(title, color=COLOR_INK, fontsize=11)
    _chart_style(ax)
    fig.tight_layout()
    return fig


def display_hw(img_01: np.ndarray) -> tuple[int, int]:
    """Height, width of image for overlay (handles CHW or HWC display arrays)."""
    a = np.asarray(img_01)
    if a.ndim == 3 and a.shape[0] == 3:
        return int(a.shape[1]), int(a.shape[2])
    if a.ndim >= 2:
        return int(a.shape[0]), int(a.shape[1])
    raise ValueError(f"Cannot get H,W from shape {a.shape}")


def _heat_hw(heat: np.ndarray, base_hw: tuple[int, int]) -> np.ndarray:
    """Force heatmap to 2D (H, W) matching the display image."""
    h = np.asarray(heat, dtype=np.float32)
    while h.ndim > 2:
        h = h.squeeze()
    if h.ndim == 1:
        raise ValueError(f"Cannot reshape 1D heat {h.shape} to {base_hw}")
    if h.shape != base_hw:
        h = np.asarray(h, dtype=np.float32).reshape(base_hw)
    return h


def _rgba_from_heat(heat_hw: np.ndarray, *, cmap_name: str) -> np.ndarray:
    norm = Normalize(vmin=0, vmax=max(float(heat_hw.max()), 1e-8))
    try:
        colormap = plt.colormaps[cmap_name]
    except (AttributeError, KeyError):
        colormap = cm.get_cmap(cmap_name)
    rgba = np.asarray(colormap(norm(heat_hw)), dtype=np.float32)
    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError(f"Expected HxWx4 colormap output, got {rgba.shape}")
    return rgba


def heatmap_rgb(heat: np.ndarray, *, cmap_name: str = "turbo", base_hw: tuple[int, int] | None = None) -> np.ndarray:
    if base_hw is None:
        h2 = np.asarray(heat, dtype=np.float32).squeeze()
        base_hw = (int(h2.shape[0]), int(h2.shape[1]))
    heat_hw = _heat_hw(heat, base_hw)
    rgba = _rgba_from_heat(heat_hw, cmap_name=cmap_name)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def composite_overlay(
    base_01: np.ndarray,
    heat: np.ndarray,
    *,
    alpha: float = 0.5,
    cmap_name: str = "turbo",
) -> np.ndarray:
    """Blend heatmap onto input image (base_01 in [0,1])."""
    base = display_to_rgb_uint8(base_01)
    if base.ndim == 2:
        base = np.stack([base, base, base], axis=-1)
    h, w = int(base.shape[0]), int(base.shape[1])

    heat_hw = _heat_hw(heat, (h, w))
    rgba = _rgba_from_heat(heat_hw, cmap_name=cmap_name)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    mask = (heat_hw > 0).astype(np.float32)
    a = (rgba[:, :, 3] * float(alpha) * mask)[:, :, np.newaxis]

    out = base.astype(np.float32) * (1.0 - a) + rgb.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def alibi_to_display(spec_name: str, img_alibi: np.ndarray) -> np.ndarray:
    return alibi_to_display_01(img_alibi)
