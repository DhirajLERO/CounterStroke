"""Metrics explorer — from-table charts (default) or preplotted JPGs; optional cross-dataset compare."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .datasets import get_app_dataset_spec
from .metrics_catalog import (
    DEFAULT_METRIC_IDS,
    METRIC_GROUPS,
    METRICS_BY_ID,
    MetricDef,
    metrics_for_group,
)
from .metrics_io import (
    COMPARE_DATASET_KEYS,
    load_all_dataset_bundles,
    load_dataset_bundle,
    methods_in_data,
    methods_union,
    pdf_path_for_stem,
    plot_path_for_stem,
    summary_table_for_metric,
)
from .metrics_paths import discover_available_files, resolve_metrics_dirs
from .metrics_plots import (
    PlotOptions,
    build_figure,
    fig_compare_datasets,
    figure_to_pdf_bytes,
    recommended_plot_kind,
)
from .path_validate import validate_metrics_paths
from .ui_nav import PAGE_HOME, go

PLOT_FROM_TABLES = "From tables"
PLOT_PREPLOTTED = "Preplotted (pipeline JPG)"
SCOPE_SINGLE = "Single dataset"
SCOPE_COMPARE = "Compare all datasets"


def _dataset_label(name: str) -> str:
    return {"mnist": "MNIST", "cifar10": "CIFAR-10", "imagenet100": "ImageNet-100"}.get(
        name, name
    )


def _render_metric_about(metric: MetricDef, dataset: str) -> None:
    st.markdown(f"**{metric.label}** — {metric.description}")
    st.markdown(f"**How it is computed:** {metric.how_computed}")
    if metric.formula:
        st.latex(metric.formula)
    dir_hint = {"lower": "Lower is better", "higher": "Higher is better", "context": "Contextual"}
    st.caption(f"Direction: {dir_hint.get(metric.direction, metric.direction)}")
    if metric.dataset_notes and dataset in metric.dataset_notes:
        st.info(metric.dataset_notes[dataset])


def _pick_metrics_ui() -> list[str]:
    st.markdown("**Metrics** (select one or more — plotted together when comparing datasets)")
    selected: list[str] = []
    for group in METRIC_GROUPS:
        with st.expander(group, expanded=group in ("Success", "Proximity", "Distribution")):
            for m in metrics_for_group(group):
                default_on = m.id in DEFAULT_METRIC_IDS
                if st.checkbox(m.label, value=default_on, key=f"metric_cb_{m.id}"):
                    selected.append(m.id)
    if not selected:
        selected = list(DEFAULT_METRIC_IDS)
    return selected


def _summary_table(metric: MetricDef, bundle, methods: list[str]) -> pd.DataFrame:
    return summary_table_for_metric(metric, bundle, methods)


def _download_pdf_button(pdf_bytes: bytes, filename: str, key: str) -> None:
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        key=key,
    )


def _render_plot_block(
    stem: str,
    results_dir: Path,
    *,
    caption: str,
    shown_stems: set[str],
    key_prefix: str = "",
) -> bool:
    if stem in shown_stems:
        return False
    img = plot_path_for_stem(results_dir, stem)
    if img is None:
        return False
    shown_stems.add(stem)
    st.image(str(img), caption=caption, width="stretch")
    pdf = pdf_path_for_stem(results_dir, stem)
    if pdf is not None:
        with open(pdf, "rb") as f:
            st.download_button(
                f"Download pipeline PDF ({stem})",
                data=f.read(),
                file_name=pdf.name,
                mime="application/pdf",
                key=f"pipe_pdf_{key_prefix}{stem}",
            )
    return True


def _chart_style_options(metric: MetricDef) -> list[str]:
    opts = ["Auto", "Bar mean±error", "Boxplot"]
    kind = recommended_plot_kind(metric)
    if kind == "twin_bar":
        opts.append("Twin bar")
    if metric.csv_source == "distribution":
        opts.append("Heatmap (all distribution metrics)")
    return opts


def _map_chart_style(label: str) -> str:
    return {
        "Auto": "auto",
        "Bar mean±error": "bar",
        "Boxplot": "boxplot",
        "Twin bar": "twin_bar",
        "Heatmap (all distribution metrics)": "heatmap",
    }.get(label, "auto")


def _render_single_metric(
    metric: MetricDef,
    *,
    dataset: str,
    bundle,
    methods: list[str],
    plot_source: str,
    chart_style_label: str,
    error_bars_label: str,
    shown_stems: set[str],
) -> None:
    st.subheader(metric.label)

    with st.expander("About this metric", expanded=False):
        _render_metric_about(metric, dataset)

    table = _summary_table(metric, bundle, methods)
    if not table.empty:
        st.dataframe(table, width="stretch", hide_index=True)
    else:
        st.caption("No numeric summary for this metric with current filters.")

    style_key = _map_chart_style(chart_style_label)
    err_key = error_bars_label.lower()

    if plot_source == PLOT_FROM_TABLES:
        opts = PlotOptions(
            chart_style=style_key,  # type: ignore[arg-type]
            error_bars=err_key if err_key in ("sem", "std", "none") else "sem",  # type: ignore[arg-type]
        )
        fig = build_figure(metric, bundle, methods, opts=opts)
        if fig is not None:
            fname = f"{dataset}_{metric.id}_{style_key}.pdf"
            pdf_bytes = figure_to_pdf_bytes(fig)
            st.pyplot(fig, clear_figure=True)
            _download_pdf_button(pdf_bytes, fname, key=f"dl_pdf_{metric.id}_{style_key}")
            plt.close(fig)
        else:
            st.warning("Could not build chart from tables (missing CSV column or data).")
    else:
        shown = False
        for stem in metric.plot_stems:
            if _render_plot_block(
                stem,
                bundle.dirs.results_dir,
                caption=f"{stem} (pipeline)",
                shown_stems=shown_stems,
            ):
                shown = True
        if not shown:
            st.caption(
                f"Pipeline JPG not found — switch to **{PLOT_FROM_TABLES}** or run metrics_and_plots."
            )

    st.divider()


def _reset_metrics_paths(dataset: str) -> None:
    """Point path session + widget keys at the dataset defaults (clears sticky MNIST paths)."""
    spec = get_app_dataset_spec(dataset)
    gen = str(spec.default_gen_root.resolve())
    res = str(spec.default_results_dir.resolve())
    st.session_state.metrics_gen_root = gen
    st.session_state.metrics_results_dir = res
    st.session_state.metrics_gen_input = gen
    st.session_state.metrics_results_input = res
    if "metrics_methods_pick" in st.session_state:
        del st.session_state.metrics_methods_pick


def _ensure_metrics_paths(dataset: str) -> None:
    if not st.session_state.get("metrics_gen_root") or not st.session_state.get("metrics_results_dir"):
        _reset_metrics_paths(dataset)


def _ensure_compare_path_defaults() -> None:
    for name in COMPARE_DATASET_KEYS:
        spec = get_app_dataset_spec(name)
        gk = f"metrics_cmp_gen_{name}"
        rk = f"metrics_cmp_res_{name}"
        if gk not in st.session_state or not st.session_state[gk]:
            st.session_state[gk] = str(spec.default_gen_root.resolve())
        if rk not in st.session_state or not st.session_state[rk]:
            st.session_state[rk] = str(spec.default_results_dir.resolve())


def _compare_overrides_from_session() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for name in COMPARE_DATASET_KEYS:
        out[name] = {
            "gen": st.session_state.get(f"metrics_cmp_gen_{name}", ""),
            "results": st.session_state.get(f"metrics_cmp_res_{name}", ""),
        }
    return out


def _render_single_dataset_controls(dataset_options: list[str]) -> str:
    loaded = st.session_state.metrics_dataset
    if loaded not in dataset_options:
        loaded = "mnist"
        st.session_state.metrics_dataset = loaded
    _ensure_metrics_paths(loaded)

    if "metrics_dataset_pick" not in st.session_state:
        st.session_state.metrics_dataset_pick = loaded

    c_ds, c_load, c_refresh = st.columns([3, 1, 1])
    with c_ds:
        pick = st.selectbox(
            "Dataset",
            dataset_options,
            format_func=_dataset_label,
            key="metrics_dataset_pick",
        )
    with c_load:
        st.write("")
        load_clicked = st.button("Load dataset", type="primary", width="stretch")
    with c_refresh:
        st.write("")
        refresh_clicked = st.button("Refresh", width="stretch")

    if load_clicked:
        st.session_state.metrics_dataset = pick
        _reset_metrics_paths(pick)
        st.rerun()

    if refresh_clicked:
        gen_now = st.session_state.get("metrics_gen_input") or st.session_state.metrics_gen_root
        res_now = st.session_state.get("metrics_results_input") or st.session_state.metrics_results_dir
        st.session_state.metrics_gen_root = gen_now
        st.session_state.metrics_results_dir = res_now
        if "metrics_methods_pick" in st.session_state:
            del st.session_state.metrics_methods_pick
        st.rerun()

    dataset = st.session_state.metrics_dataset
    if pick != dataset:
        st.warning(
            f"Selected **{_dataset_label(pick)}** but still showing **{_dataset_label(dataset)}**. "
            "Click **Load dataset** to switch."
        )
    else:
        st.caption(f"Loaded: **{_dataset_label(dataset)}**")

    spec = get_app_dataset_spec(dataset)
    with st.expander("Paths (defaults + optional override)", expanded=False):
        gen_val = st.text_input("Generation root (CSVs)", key="metrics_gen_input")
        res_val = st.text_input(
            "Results dir (plots + summary_tables.csv)",
            key="metrics_results_input",
        )
        st.session_state.metrics_gen_root = gen_val
        st.session_state.metrics_results_dir = res_val

        p1, p2 = st.columns(2)
        with p1:
            if st.button("Reset paths to dataset defaults"):
                _reset_metrics_paths(dataset)
                st.rerun()
        with p2:
            if st.button("Check paths"):
                dirs = resolve_metrics_dirs(spec, gen_override=gen_val, results_override=res_val)
                checks = validate_metrics_paths(dirs)
                for name, (ok, msg) in checks.items():
                    if ok:
                        st.success(f"**{name}**: {msg}")
                    else:
                        st.error(f"**{name}**: {msg}")

    return dataset


def _render_compare_path_controls() -> None:
    _ensure_compare_path_defaults()
    with st.expander("Paths per dataset (optional override)", expanded=False):
        for name in COMPARE_DATASET_KEYS:
            st.markdown(f"**{_dataset_label(name)}**")
            st.text_input(
                f"Generation root — {_dataset_label(name)}",
                key=f"metrics_cmp_gen_{name}",
            )
            st.text_input(
                f"Results dir — {_dataset_label(name)}",
                key=f"metrics_cmp_res_{name}",
            )
        if st.button("Reset all compare paths to defaults"):
            for name in COMPARE_DATASET_KEYS:
                spec = get_app_dataset_spec(name)
                st.session_state[f"metrics_cmp_gen_{name}"] = str(spec.default_gen_root.resolve())
                st.session_state[f"metrics_cmp_res_{name}"] = str(spec.default_results_dir.resolve())
            if "metrics_methods_pick" in st.session_state:
                del st.session_state.metrics_methods_pick
            st.rerun()


def _render_compare_from_tables(
    bundles: dict,
    selected_metrics: list[str],
    methods: list[str],
    error_bars_label: str,
) -> None:
    available = {
        k: b
        for k, b in bundles.items()
        if discover_available_files(b.dirs)["gen_root_exists"]
    }
    missing = [k for k in COMPARE_DATASET_KEYS if k not in available]
    if missing:
        st.caption(
            "Skipping missing gen roots: "
            + ", ".join(_dataset_label(m) for m in missing)
        )
    if not available:
        st.error("No dataset generation roots found. Check paths under the expander.")
        return

    err_key = error_bars_label.lower()
    err: str = err_key if err_key in ("sem", "std", "none") else "sem"

    st.markdown("### Cross-dataset group plot")
    fig = fig_compare_datasets(
        selected_metrics,
        available,
        methods,
        error_bars=err,  # type: ignore[arg-type]
    )
    if fig is None:
        st.warning("Could not build cross-dataset chart (no metrics/methods/data).")
        return
    pdf_bytes = figure_to_pdf_bytes(fig)
    st.pyplot(fig, clear_figure=True)
    mid_tag = "_".join(selected_metrics[:4])
    _download_pdf_button(
        pdf_bytes,
        f"compare_{mid_tag}.pdf",
        key=f"dl_pdf_compare_{mid_tag}_{err}",
    )
    plt.close(fig)

    with st.expander("Per-dataset summary tables", expanded=False):
        for ds, bundle in available.items():
            st.markdown(f"#### {_dataset_label(ds)}")
            for mid in selected_metrics:
                metric = METRICS_BY_ID.get(mid)
                if metric is None:
                    continue
                table = _summary_table(metric, bundle, methods)
                st.caption(metric.label)
                if not table.empty:
                    st.dataframe(table, width="stretch", hide_index=True)
                else:
                    st.caption("No rows.")


def _render_compare_preplotted(bundles: dict, selected_metrics: list[str]) -> None:
    st.markdown("### Preplotted pipeline figures")
    for ds, bundle in bundles.items():
        discovery = discover_available_files(bundle.dirs)
        with st.expander(_dataset_label(ds), expanded=ds == "mnist"):
            if not discovery["results_dir_exists"]:
                st.warning(f"Results dir missing: `{bundle.dirs.results_dir}`")
                continue
            shown_stems: set[str] = set()
            any_shown = False
            for mid in selected_metrics:
                metric = METRICS_BY_ID.get(mid)
                if metric is None:
                    continue
                st.markdown(f"**{metric.label}**")
                for stem in metric.plot_stems:
                    if _render_plot_block(
                        stem,
                        bundle.dirs.results_dir,
                        caption=f"{_dataset_label(ds)} · {stem}",
                        shown_stems=shown_stems,
                        key_prefix=f"{ds}_",
                    ):
                        any_shown = True
            if not any_shown:
                st.caption("No pipeline JPGs found for selected metrics.")


def render_metrics() -> None:
    st.title("Evaluation metrics")
    st.caption(
        "Plot from CSV tables by default, or view preplotted pipeline JPGs. "
        "Compare all three datasets with grouped multi-bar charts."
    )

    if st.button("← Home"):
        go(PAGE_HOME)

    dataset_options = list(COMPARE_DATASET_KEYS)

    sc1, sc2 = st.columns(2)
    with sc1:
        scope = st.radio(
            "Scope",
            [SCOPE_SINGLE, SCOPE_COMPARE],
            horizontal=True,
            key="metrics_scope",
        )
    with sc2:
        if "metrics_plot_source" not in st.session_state:
            st.session_state.metrics_plot_source = PLOT_FROM_TABLES
        # Migrate old radio values
        if st.session_state.metrics_plot_source in ("Pipeline JPG", "Interactive"):
            st.session_state.metrics_plot_source = (
                PLOT_PREPLOTTED
                if st.session_state.metrics_plot_source == "Pipeline JPG"
                else PLOT_FROM_TABLES
            )
        plot_source = st.radio(
            "Plot source",
            [PLOT_FROM_TABLES, PLOT_PREPLOTTED],
            horizontal=True,
            key="metrics_plot_source",
        )

    from_tables = plot_source == PLOT_FROM_TABLES

    tb1, tb2 = st.columns(2)
    with tb1:
        chart_style_label = st.selectbox(
            "Chart style (single-dataset, From tables)",
            ["Auto", "Bar mean±error", "Boxplot", "Twin bar", "Heatmap (all distribution metrics)"],
            index=0,
            key="metrics_chart_style",
            disabled=not from_tables or scope == SCOPE_COMPARE,
        )
    with tb2:
        error_bars_label = st.selectbox(
            "Error bars",
            ["SEM", "Std", "None"],
            index=0,
            key="metrics_error_bars",
            disabled=not from_tables,
        )

    selected_metrics = _pick_metrics_ui()

    if scope == SCOPE_COMPARE:
        _render_compare_path_controls()
        bundles = load_all_dataset_bundles(_compare_overrides_from_session())
        all_methods = methods_union(bundles)
        method_pick = st.multiselect(
            "Methods",
            all_methods,
            default=all_methods,
            key="metrics_methods_pick",
        )
        methods = method_pick or all_methods

        st.markdown("---")
        if from_tables:
            _render_compare_from_tables(bundles, selected_metrics, methods, error_bars_label)
        else:
            _render_compare_preplotted(bundles, selected_metrics)
        return

    dataset = _render_single_dataset_controls(dataset_options)
    spec = get_app_dataset_spec(dataset)
    bundle = load_dataset_bundle(
        spec,
        gen_override=st.session_state.metrics_gen_root or None,
        results_override=st.session_state.metrics_results_dir or None,
    )
    discovery = discover_available_files(bundle.dirs)

    if not discovery["gen_root_exists"]:
        st.error(
            f"Generation root not found: `{bundle.dirs.gen_root}`. "
            f"Run `{spec.dataset_dir.name}/metrics_and_plots.py --gen-root …` on the GPU machine."
        )
        return

    if not discovery["gen_csvs"].get("per_sample_metrics.csv"):
        st.warning(
            f"`per_sample_metrics.csv` missing under `{bundle.dirs.gen_root}`. "
            "Generate metrics with the dataset pipeline before browsing."
        )

    all_methods = methods_in_data(bundle.df_valid)
    if not all_methods:
        all_methods = methods_in_data(bundle.frames.get("distribution"))

    method_pick = st.multiselect(
        "Methods",
        all_methods,
        default=all_methods,
        key="metrics_methods_pick",
    )
    methods = method_pick or all_methods

    st.markdown("---")
    st.markdown("### Results")
    shown_stems: set[str] = set()
    for mid in selected_metrics:
        metric = METRICS_BY_ID.get(mid)
        if metric is None:
            continue
        per_metric_style = chart_style_label
        if from_tables:
            allowed = _chart_style_options(metric)
            if chart_style_label not in allowed and chart_style_label != "Auto":
                per_metric_style = "Auto"
        _render_single_metric(
            metric,
            dataset=dataset,
            bundle=bundle,
            methods=methods,
            plot_source=plot_source,
            chart_style_label=per_metric_style,
            error_bars_label=error_bars_label,
            shown_stems=shown_stems,
        )
