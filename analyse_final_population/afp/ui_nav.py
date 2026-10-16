"""Session navigation between app pages."""

from __future__ import annotations

import streamlit as st

PAGE_HOME = "home"
PAGE_DATASET = "dataset"
PAGE_SETUP = "setup"
PAGE_ANALYSIS = "analysis"
PAGE_METRICS = "metrics"


def init_session() -> None:
    if "page" not in st.session_state:
        st.session_state.page = PAGE_HOME
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "cfg" not in st.session_state:
        st.session_state.cfg = None
    if "paths_validated" not in st.session_state:
        st.session_state.paths_validated = False
    if "metrics_dataset" not in st.session_state:
        st.session_state.metrics_dataset = "mnist"
    if "metrics_gen_root" not in st.session_state:
        st.session_state.metrics_gen_root = ""
    if "metrics_results_dir" not in st.session_state:
        st.session_state.metrics_results_dir = ""


def go(page: str) -> None:
    st.session_state.page = page
    st.rerun()


def breadcrumb(dataset: str | None = None) -> None:
    parts = ["Home"]
    ds = dataset or (
        st.session_state.get("metrics_dataset")
        if st.session_state.page == PAGE_METRICS
        else None
    )
    if ds:
        label = {"mnist": "MNIST", "cifar10": "CIFAR-10", "imagenet100": "ImageNet-100"}.get(
            ds, ds
        )
        parts.append(label)
    if st.session_state.page == PAGE_SETUP:
        parts.append("Setup")
    elif st.session_state.page == PAGE_ANALYSIS:
        parts.append("Analysis")
    elif st.session_state.page == PAGE_METRICS:
        parts.append("Metrics")
    st.markdown(
        " › ".join(f"**{p}**" if i == len(parts) - 1 else p for i, p in enumerate(parts)),
    )
