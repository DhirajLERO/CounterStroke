"""UI constants and Streamlit width helpers."""

from __future__ import annotations

GITHUB_URL = "https://github.com/your-org/GracoImage"

# Pastel theme tokens
COLOR_LAVENDER = "#b8a9e8"
COLOR_MINT = "#9ee7c4"
COLOR_PEACH = "#ffc9a8"
COLOR_INK = "#3d3a5c"
COLOR_MUTED = "#6b6888"

DEFAULT_MAP_WIDTH = 480
DEFAULT_OVERLAY_ALPHA = 0.55
DEFAULT_COCHANGE_TAU = 0.5

CHANGE_MAP_METHOD_LABELS: dict[str, str] = {
    "sum_intensity": "Sum intensity (normalized) — total change magnitude per pixel",
    "mean_intensity": "Mean intensity — average change among CFs that edited each pixel",
    "change_frequency": "Change frequency — how often each pixel was edited",
    "max_intensity": "Max intensity — strongest single CF change per pixel",
    "cochange_strict": "Co-change — pixels edited by most CFs (consensus)",
    "spatial_coherence": "Spatial coherence — frequency map smoothed to show regions",
}

CHANGE_MAP_METHOD_SHORT: dict[str, str] = {
    "sum_intensity": "Sum intensity",
    "mean_intensity": "Mean intensity",
    "change_frequency": "Change frequency",
    "max_intensity": "Max intensity",
    "cochange_strict": "Co-change",
    "spatial_coherence": "Spatial coherence",
}

ALL_MAP_METHODS: list[str] = list(CHANGE_MAP_METHOD_SHORT.keys())

ATTRIBUTION_METHOD_SHORT: dict[str, str] = {
    "integrated_gradients": "Integrated Gradients",
    "input_x_gradient": "Input × Gradient",
    "smoothgrad": "SmoothGrad",
    "occlusion": "Occlusion",
    "gradcam": "Grad-CAM",
    "shap_gradient": "SHAP (gradient)",
}

ALL_ATTRIBUTION_METHODS: list[str] = list(ATTRIBUTION_METHOD_SHORT.keys())


def button_width(stretch: bool = True) -> str:
    return "stretch" if stretch else "content"


def image_width_pixels(pixels: int) -> int:
    return int(pixels)
