"""Home page — Graco method overview."""

from __future__ import annotations

import streamlit as st

from .config_ui import GITHUB_URL
from .ui_nav import PAGE_DATASET, PAGE_METRICS, go


def render_home() -> None:
    st.markdown(
        """
<div class="paper-card">
<h1 style="margin-top:0;color:#3d3a5c;">Graco Counterfactuals via Turtle Programs</h1>
<p class="abstract"><strong>Abstract.</strong> Graco evolves executable <em>turtle graphics</em>
phenotypes on an input image. A trained classifier scores each candidate; multi-objective evolution
favors label flips with sparse, localized edits (proximity and sparsity objectives O2/O3).</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Method")
    st.markdown(
        """
1. **Representation** — Each individual is a program (phenotype) that draws strokes on the image
   using a turtle grammar (GE / NSGA-II).
2. **Fitness** — Minimize conflict with the desired class probability while keeping edits small
   and close to the original (or blank canvas).
3. **Final generation** — Hundreds of individuals are saved per run (`pred_class`, `pred_prob`,
   phenotype, optional rendered image in pickle).
4. **This app** — Explores *where* the final population lands in label space and *where* pixels
   change in aggregate (multiple heatmap methods on the **input image** as background).
        """
    )

    st.markdown("### Experiment modes")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**Factual images** — 100 eval samples per dataset; one Graco run per "
            "`{index}_label_{y}` folder."
        )
    with col2:
        st.info(
            "**Blank canvas** — White/black canvas evolved toward each target class; "
            "belief landscape per `target_XXX` CSV."
        )

    st.link_button("GitHub (placeholder)", GITHUB_URL, type="secondary")

    st.divider()
    st.markdown("### Evaluation metrics")
    st.markdown(
        """
Browse **precomputed counterfactual metrics** (proximity, classifier scores, FID, CMMD, …)
and pipeline plots for MNIST, CIFAR-10, and ImageNet-100. Reads CSVs under
`outputs/generation/` and figures under `outputs/results/generation/`.
        """
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("View metrics →", type="primary", width="stretch"):
            go(PAGE_METRICS)
    with col_b:
        if st.button("Explore populations →", type="secondary", width="stretch"):
            go(PAGE_DATASET)
