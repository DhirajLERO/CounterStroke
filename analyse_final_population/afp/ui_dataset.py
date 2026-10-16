"""Dataset selection cards."""

from __future__ import annotations

import streamlit as st

from .ui_nav import PAGE_SETUP, go

_DATASETS = [
    ("mnist", "MNIST", "28×28 grayscale · black blank", "#e8dff5"),
    ("cifar10", "CIFAR-10", "32×32 RGB · white blank", "#dff5ec"),
    ("imagenet100", "ImageNet-100", "299×299 RGB · 100 classes", "#fff0e0"),
]


def render_dataset_picker() -> None:
    st.subheader("Choose a dataset")
    st.caption("You will confirm result directories on the next step.")

    cols = st.columns(3)
    for col, (key, title, desc, _bg) in zip(cols, _DATASETS):
        with col:
            st.markdown(f'<div class="dataset-card">', unsafe_allow_html=True)
            st.markdown(f"#### {title}")
            st.caption(desc)
            if st.button(f"Select {title}", key=f"ds_{key}", width="stretch"):
                st.session_state.dataset = key
                go(PAGE_SETUP)
            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("← Back to home"):
        from .ui_nav import PAGE_HOME

        go(PAGE_HOME)
