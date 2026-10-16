"""Attribution baseline heatmaps (Grad-CAM, IG, …) for factual Analyse."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from .attribution_maps import (
    ALL_BASELINE_METHOD_IDS,
    DEFAULT_ATTRIBUTION_METHODS,
    resolve_baseline_maps,
)
from .config_ui import (
    ATTRIBUTION_METHOD_SHORT,
    DEFAULT_MAP_WIDTH,
    image_width_pixels,
)
from .datasets import AppDatasetSpec
from .ui_change_map import _render_image_for_view
from .ui_image_download import show_downloadable_image
from .viz import display_hw


def _pick_attribution_methods_ui(key_prefix: str) -> list[str]:
    st.markdown("**Attribution methods** (select one or more to compare)")
    state_key = f"{key_prefix}_methods_selected"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(DEFAULT_ATTRIBUTION_METHODS)

    cols = st.columns(3)
    selected: list[str] = []
    for i, method_id in enumerate(ALL_BASELINE_METHOD_IDS):
        short = ATTRIBUTION_METHOD_SHORT.get(method_id, method_id)
        with cols[i % 3]:
            if st.checkbox(
                short,
                value=method_id in st.session_state[state_key],
                key=f"{key_prefix}_cb_{method_id}",
            ):
                selected.append(method_id)

    if not selected:
        st.warning("Select at least one attribution method.")
        selected = list(DEFAULT_ATTRIBUTION_METHODS)

    st.session_state[state_key] = selected
    return selected


def render_attribution_baselines_panel(
    *,
    spec: AppDatasetSpec,
    sample_idx: int,
    orig_alibi,
    base_display_01,
    target_pred_class: int,
    cfg: dict,
    title: str,
    key_prefix: str = "attr",
) -> None:
    st.subheader(title)
    st.caption(
        "Standard XAI baselines on the factual input for the flip/top class. "
        "Loads `explainability/outputs/.../maps/{method}.npz` when present; "
        "otherwise computes and caches."
    )

    weights = Path(cfg["weights"]) if cfg.get("weights") else None
    if weights is None or not weights.is_file():
        st.warning("Classifier weights missing — attribution baselines disabled.")
        return

    selected_methods = _pick_attribution_methods_ui(key_prefix)

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

    with st.spinner("Resolving attribution maps (load or compute)…"):
        results = resolve_baseline_maps(
            dataset=spec.name,
            sample_idx=sample_idx,
            methods=selected_methods,
            img_alibi=orig_alibi,
            target_class=target_pred_class,
            weights=weights,
            device=cfg.get("device", "cpu"),
        )

    n_ok = sum(1 for r in results.values() if r.importance is not None)
    sources = {r.source for r in results.values() if r.importance is not None}
    source_note = ", ".join(sorted(sources)) if sources else "none"
    st.caption(
        f"**{n_ok}/{len(selected_methods)}** maps · target class **{target_pred_class}** · "
        f"source: **{source_note}**"
    )

    if n_ok == 0:
        errors = [f"`{m}`: {r.error}" for m, r in results.items() if r.error]
        st.warning("No attribution maps available.")
        for e in errors:
            st.caption(e)
        return

    base_hw = display_hw(base_display_01)
    n_cols = min(3, len(selected_methods))
    rows = [selected_methods[i : i + n_cols] for i in range(0, len(selected_methods), n_cols)]

    for row_methods in rows:
        cols = st.columns(len(row_methods))
        for col, method_id in zip(cols, row_methods):
            with col:
                short = ATTRIBUTION_METHOD_SHORT.get(method_id, method_id)
                res = results[method_id]
                if res.importance is None:
                    st.error(f"{short}")
                    st.caption(res.error or "failed")
                    continue
                img, cap = _render_image_for_view(
                    base_display_01,
                    res.importance,
                    view=view,
                    intensity=intensity,
                    cmap=cmap,
                    base_hw=base_hw,
                )
                show_downloadable_image(
                    img,
                    caption=f"{short} · {cap} · {res.source}",
                    width=image_width_pixels(width),
                    key=f"{key_prefix}_{method_id}",
                    filename_stem=f"{key_prefix}_{method_id}",
                )
