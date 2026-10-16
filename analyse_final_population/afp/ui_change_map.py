"""Change map display widget with multi-method comparison."""

from __future__ import annotations

import streamlit as st

from .analysis import ChangeMapMethod, collect_diffs_for_target, heat_from_diffs
from .config_ui import (
    ALL_MAP_METHODS,
    CHANGE_MAP_METHOD_SHORT,
    DEFAULT_MAP_WIDTH,
    image_width_pixels,
)
from .datasets import AppDatasetSpec
from .inputs import display_to_rgb_uint8
from .population_io import IndividualRecord
from .ui_image_download import show_downloadable_image
from .viz import composite_overlay, display_hw, heatmap_rgb


def _render_image_for_view(
    base_display_01,
    heat,
    *,
    view: str,
    intensity: float,
    cmap: str,
    base_hw: tuple[int, int],
):
    if view == "Input only":
        return display_to_rgb_uint8(base_display_01), "Input only"
    if view == "Heatmap only":
        return heatmap_rgb(heat, cmap_name=cmap, base_hw=base_hw), "Heatmap only"
    return (
        composite_overlay(base_display_01, heat, alpha=intensity, cmap_name=cmap),
        "Overlay",
    )


def _pick_methods_ui(key_prefix: str, cfg: dict) -> list[str]:
    """Checkboxes to select one or more aggregation methods."""
    st.markdown("**Mapping methods** (select one or more to compare)")
    state_key = f"{key_prefix}_methods_selected"
    if state_key not in st.session_state:
        default = cfg.get("map_methods") or [cfg.get("map_method", "sum_intensity")]
        if isinstance(default, str):
            default = [default]
        st.session_state[state_key] = [m for m in default if m in ALL_MAP_METHODS] or ["sum_intensity"]

    cols = st.columns(3)
    selected: list[str] = []
    for i, method_id in enumerate(ALL_MAP_METHODS):
        short = CHANGE_MAP_METHOD_SHORT.get(method_id, method_id)
        with cols[i % 3]:
            if st.checkbox(
                short,
                value=method_id in st.session_state[state_key],
                key=f"{key_prefix}_cb_{method_id}",
            ):
                selected.append(method_id)

    if not selected:
        st.warning("Select at least one mapping method.")
        selected = ["sum_intensity"]

    st.session_state[state_key] = selected
    return selected


def render_change_map_panel(
    *,
    spec: AppDatasetSpec,
    orig_alibi,
    base_display_01,
    records: list[IndividualRecord],
    target_pred_class: int,
    cfg: dict,
    title: str,
    key_prefix: str = "cm",
    download_stem_fn=None,
) -> None:
    st.subheader(title)

    selected_methods = _pick_methods_ui(key_prefix, cfg)

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        view = st.radio(
            "View",
            ["Overlay", "Input only", "Heatmap only"],
            horizontal=True,
            key=f"{key_prefix}_view",
        )
    with r1c2:
        intensity = st.slider(
            "Overlay intensity",
            0.0,
            1.0,
            float(cfg.get("overlay_alpha", 0.55)),
            0.05,
            key=f"{key_prefix}_alpha",
        )
    with r1c3:
        width = st.slider(
            "Width per panel (px)",
            160,
            480,
            int(cfg.get("map_width", DEFAULT_MAP_WIDTH)),
            20,
            key=f"{key_prefix}_width",
        )

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        cmap = st.selectbox(
            "Colormap",
            ["turbo", "magma", "plasma", "viridis", "coolwarm"],
            index=["turbo", "magma", "plasma", "viridis", "coolwarm"].index(
                cfg.get("cmap_name", "turbo")
            )
            if cfg.get("cmap_name", "turbo") in ("turbo", "magma", "plasma", "viridis", "coolwarm")
            else 0,
            key=f"{key_prefix}_cmap",
        )
    with r2c2:
        cochange_tau = st.slider(
            "Co-change τ (for Co-change method)",
            0.1,
            1.0,
            float(cfg.get("cochange_tau", 0.5)),
            0.05,
            key=f"{key_prefix}_tau",
            disabled=ChangeMapMethod.COCHANGE_STRICT.value not in selected_methods,
        )

    with st.spinner("Rendering counterfactuals (once)…"):
        diffs, n_used, shape = collect_diffs_for_target(
            spec,
            orig_alibi,
            records,
            target_pred_class=target_pred_class,
            max_render=int(cfg.get("max_render", 200)),
        )

    st.caption(
        f"**{n_used}** counterfactuals · target class **{target_pred_class}** · "
        f"comparing **{len(selected_methods)}** method(s) on the same input background"
    )

    if n_used == 0:
        st.warning("No counterfactuals to render for this class.")
        return

    base_hw = display_hw(base_display_01)
    n_cols = min(3, len(selected_methods))
    rows = [selected_methods[i : i + n_cols] for i in range(0, len(selected_methods), n_cols)]

    for row_methods in rows:
        cols = st.columns(len(row_methods))
        for col, method_id in zip(cols, row_methods):
            with col:
                heat = heat_from_diffs(
                    diffs,
                    shape,
                    method_id,
                    cochange_tau=cochange_tau,
                )
                img, cap = _render_image_for_view(
                    base_display_01,
                    heat,
                    view=view,
                    intensity=intensity,
                    cmap=cmap,
                    base_hw=base_hw,
                )
                short = CHANGE_MAP_METHOD_SHORT.get(method_id, method_id)
                if download_stem_fn is not None:
                    stem = download_stem_fn(method_id)
                else:
                    stem = f"{key_prefix}_{method_id}"
                show_downloadable_image(
                    img,
                    caption=f"{short} · {cap}",
                    width=image_width_pixels(width),
                    key=f"{key_prefix}_{method_id}",
                    filename_stem=stem,
                )
