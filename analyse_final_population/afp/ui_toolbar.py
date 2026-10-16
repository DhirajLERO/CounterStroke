"""Top toolbar with settings popovers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from .datasets import resolve_blank_graco_root, resolve_factual_graco_root
from .ui_nav import PAGE_DATASET, PAGE_SETUP, go


def _resolve_device(choice: str) -> str:
    if choice in ("cuda", "auto"):
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def render_toolbar(cfg: dict) -> dict:
    """Render toolbar; return updated cfg dict."""
    spec = cfg["spec"]
    cols = st.columns([1, 1, 1, 1, 2])

    with cols[0]:
        with st.popover("📊 Dataset"):
            st.write(f"**{cfg['dataset']}**")
            if st.button("Change dataset"):
                go(PAGE_DATASET)

    with cols[1]:
        with st.popover("📁 Paths"):
            factual = st.text_input("Factual Graco", str(cfg["factual_graco"]), key="tb_factual")
            blank = st.text_input("Blank root", str(cfg["blank_graco"]), key="tb_blank")
            manifest = st.text_input("Manifest", str(cfg["manifest_path"]), key="tb_manifest")
            data_root = st.text_input("Data root", str(cfg["data_root"]), key="tb_data")
            weights = st.text_input("Weights", str(cfg["weights"]), key="tb_weights")
            if st.button("Re-validate paths"):
                go(PAGE_SETUP)
            cfg["factual_graco"] = resolve_factual_graco_root(spec, Path(factual))
            cfg["blank_graco"] = resolve_blank_graco_root(spec, Path(blank))
            cfg["manifest_path"] = Path(manifest)
            cfg["data_root"] = Path(data_root)
            cfg["weights"] = Path(weights)

    with cols[2]:
        with st.popover("⚙️ Device"):
            dev_choice = st.selectbox(
                "Device",
                ["auto", "cpu", "cuda"],
                index=["auto", "cpu", "cuda"].index(
                    cfg.get("device_choice", "auto")
                )
                if cfg.get("device_choice", "auto") in ("auto", "cpu", "cuda")
                else 0,
                key="tb_device",
            )
            cfg["device_choice"] = dev_choice
            cfg["device"] = _resolve_device(dev_choice)
            cfg["run_id"] = int(
                st.number_input("Run ID", min_value=0, value=int(cfg["run_id"]), key="tb_run")
            )

    with cols[3]:
        with st.popover("🔍 Filter"):
            cfg["prob_threshold"] = st.slider(
                "Min pred prob",
                0.0,
                1.0,
                float(cfg.get("prob_threshold", 0.5)),
                0.05,
                key="tb_prob",
            )
            cfg["max_render"] = int(
                st.slider(
                    "Max CF renders",
                    20,
                    500,
                    int(cfg.get("max_render", 200)),
                    20,
                    key="tb_maxrender",
                )
            )

    with cols[4]:
        st.caption(f"Device: `{cfg['device']}` · Run `{cfg['run_id']}` · Map methods in change-map panel")

    return cfg
