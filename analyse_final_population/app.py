#!/usr/bin/env python3
"""Streamlit app: Graco final-population explorer."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from afp.path_setup import setup_import_paths

setup_import_paths()

import matplotlib.pyplot as plt
import streamlit as st

from afp.ui_analysis import render_analysis
from afp.ui_copy import render_home
from afp.ui_dataset import render_dataset_picker
from afp.ui_metrics import render_metrics
from afp.ui_nav import (
    PAGE_ANALYSIS,
    PAGE_DATASET,
    PAGE_HOME,
    PAGE_METRICS,
    PAGE_SETUP,
    breadcrumb,
    init_session,
)
from afp.ui_setup import render_setup
from afp.viz import inject_custom_css

plt.style.use("default")


def main() -> None:
    st.set_page_config(
        page_title="Graco Final Population",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(inject_custom_css(), unsafe_allow_html=True)
    init_session()

    with st.sidebar:
        st.markdown("### Navigation")
        page = st.session_state.page
        st.caption(f"Step: **{page}**")
        if st.button("Home"):
            st.session_state.page = PAGE_HOME
            st.rerun()

    page = st.session_state.page

    if page == PAGE_HOME:
        render_home()
        return

    if page == PAGE_DATASET:
        breadcrumb()
        render_dataset_picker()
        return

    if page == PAGE_SETUP:
        breadcrumb(st.session_state.get("dataset"))
        render_setup()
        return

    if page == PAGE_ANALYSIS:
        cfg = st.session_state.get("cfg")
        if not cfg:
            st.warning("Complete setup first.")
            st.session_state.page = PAGE_SETUP
            st.rerun()
            return
        breadcrumb(cfg.get("dataset"))
        render_analysis(cfg)
        return

    if page == PAGE_METRICS:
        breadcrumb()
        render_metrics()
        return

    st.session_state.page = PAGE_HOME
    st.rerun()


if __name__ == "__main__":
    main()
