"""Path confirmation wizard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from .datasets import (
    get_app_dataset_spec,
    resolve_blank_graco_root,
    resolve_factual_graco_root,
)
from .path_validate import validate_paths
from .ui_nav import PAGE_ANALYSIS, PAGE_DATASET, go


def _build_cfg(spec, *, factual_s, blank_s, manifest_s, data_s, weights_s, run_id, class_map_s) -> dict:
    return {
        "dataset": spec.name,
        "spec": spec,
        "manifest_path": Path(manifest_s),
        "data_root": Path(data_s),
        "weights": Path(weights_s),
        "factual_graco": resolve_factual_graco_root(spec, Path(factual_s)),
        "blank_graco": resolve_blank_graco_root(spec, Path(blank_s)),
        "run_id": int(run_id),
        "max_render": 200,
        "prob_threshold": 0.5,
        "class_map": Path(class_map_s) if class_map_s else None,
        "map_method": "sum_intensity",
        "cmap_name": "turbo",
    }


def render_setup() -> None:
    dataset = st.session_state.get("dataset")
    if not dataset:
        go(PAGE_DATASET)
        return

    spec = get_app_dataset_spec(dataset)
    st.subheader(f"Confirm paths — {dataset}")

    factual_graco = st.text_input(
        "Factual Graco results",
        str(spec.default_factual_graco_root),
        key="setup_factual",
    )
    blank_graco = st.text_input(
        "Blank analysis root",
        str(spec.default_blank_graco_root),
        key="setup_blank",
    )
    manifest_path = st.text_input("Eval manifest", str(spec.default_manifest), key="setup_manifest")
    data_root = st.text_input("Data root", str(spec.default_data_root), key="setup_data")
    weights = st.text_input("Classifier weights", str(spec.default_weights), key="setup_weights")
    run_id = st.number_input("Run ID", min_value=0, value=0, step=1, key="setup_run")

    class_map_s = ""
    if dataset == "imagenet100":
        class_map_s = st.text_input(
            "Class map",
            str(spec.dataset_dir / "data_subset" / "imagenet100_class_map.json"),
            key="setup_classmap",
        )

    cfg_paths = _build_cfg(
        spec,
        factual_s=factual_graco,
        blank_s=blank_graco,
        manifest_s=manifest_path,
        data_s=data_root,
        weights_s=weights,
        run_id=run_id,
        class_map_s=class_map_s,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Test paths", width="stretch"):
            st.session_state.paths_validated = True
    with c2:
        if st.button("← Change dataset", width="stretch"):
            go(PAGE_DATASET)
    with c3:
        continue_ok = st.button("Continue to analysis →", type="primary", width="stretch")

    if st.session_state.get("paths_validated"):
        checks = validate_paths(
            spec,
            factual_graco=cfg_paths["factual_graco"],
            blank_graco=cfg_paths["blank_graco"],
            manifest_path=cfg_paths["manifest_path"],
            data_root=cfg_paths["data_root"],
            weights=cfg_paths["weights"],
            run_id=cfg_paths["run_id"],
        )
        for key, (ok, msg) in checks.items():
            if ok:
                st.success(f"**{key}**: {msg}")
            else:
                st.warning(f"**{key}**: {msg}")

    if continue_ok:
        if not cfg_paths["factual_graco"].is_dir():
            st.error("Factual Graco root must exist before continuing.")
        else:
            import torch

            cfg_paths["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            st.session_state.cfg = cfg_paths
            go(PAGE_ANALYSIS)
