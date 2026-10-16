"""Analysis workspace — factual and blank tabs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from .analysis import analyse_factual_distribution
from .blank_belief import build_belief_landscape, load_target_drilldown
from .blank_canvas import blank_alibi_for_dataset
from .blank_paths import load_summary_blank_pred
from .blank_prob_bins import pick_records_by_prob_bins
from .blank_download_names import blank_cf_stem, blank_map_stem, blank_ssim_stem, dataset_stem
from .blank_ssim_refs import best_ssim_ref, cached_val_by_class
from .class_names import get_class_names, label_name
from .classifier import cf_01_to_alibi, predict_blank, predict_probs
from .config_ui import button_width, image_width_pixels
from .generation_cfs import (
    BASELINE_METHODS,
    GRACO_METHODS,
    generation_sample_dir,
    load_method_cfs,
    load_sample_meta,
    method_caption,
)
from .graco_paths import find_graco_sample_dir
from .graco_shared import load_graco_blank_evolve, load_graco_mine
from .inputs import alibi_to_display_01, display_to_rgb_uint8, load_factual_alibi
from .manifest import ManifestEntry, load_eval_manifest
from .population_io import IndividualRecord, load_final_population
from .render import record_to_alibi
from .ui_attribution import render_attribution_baselines_panel
from .ui_change_map import render_change_map_panel
from .ui_image_download import show_downloadable_image
from .ui_toolbar import render_toolbar
from .viz import (
    fig_class_distribution,
    fig_heatmap_landscape,
    fig_o1_histogram,
    fig_softmax_bars,
)


@st.cache_data(show_spinner=False)
def _cached_manifest(path_str: str):
    return load_eval_manifest(Path(path_str))


@st.cache_data(show_spinner=False)
def _cached_factual_image(dataset: str, data_root: str, sample_idx: int, y_true: int, split: str):
    from .datasets import get_app_dataset_spec

    spec = get_app_dataset_spec(dataset)
    entry = ManifestEntry(sample_idx=sample_idx, y_true=y_true, split=split, folder_idx=0)
    alibi = load_factual_alibi(spec, entry, Path(data_root))
    return alibi, alibi_to_display_01(alibi)


@st.cache_data(show_spinner=False)
def _cached_population(graco_root: str, sample_idx: int, y_true: int, run_id: int):
    sample_dir = find_graco_sample_dir(Path(graco_root), sample_idx, y_true)
    if sample_dir is None:
        return [], "none", None
    records, src = load_final_population(sample_dir, run_id=run_id)
    return records, src, str(sample_dir)


def _show_image_row(
    items: list[tuple[np.ndarray, str, str, str]],
    *,
    max_cols: int = 6,
    width: int = 100,
) -> None:
    """Render downloadable images in rows. Each item: (rgb_uint8, caption, key, filename_stem)."""
    if not items:
        st.caption("No images to show.")
        return
    for start in range(0, len(items), max_cols):
        chunk = items[start : start + max_cols]
        cols = st.columns(len(chunk))
        for col, (rgb, caption, key, stem) in zip(cols, chunk):
            with col:
                show_downloadable_image(
                    rgb,
                    caption=caption,
                    width=image_width_pixels(width),
                    key=key,
                    filename_stem=stem,
                )


def _render_method_cf_gallery(
    spec,
    sample_idx: int,
    *,
    key_prefix: str,
) -> None:
    st.markdown("**Method counterfactuals** (from `outputs/generation/`)")
    cfs = load_method_cfs(spec, sample_idx)
    meta = load_sample_meta(generation_sample_dir(spec, sample_idx))
    if all(v is None for v in cfs.values()):
        st.caption(
            f"No `tensors.npz` under `{spec.default_gen_root}` for sample {sample_idx}."
        )
        return

    def _items(methods: tuple[str, ...]) -> list[tuple[np.ndarray, str, str, str]]:
        out: list[tuple[np.ndarray, str, str, str]] = []
        for m in methods:
            disp = cfs.get(m)
            if disp is None:
                continue
            out.append(
                (
                    display_to_rgb_uint8(disp),
                    method_caption(m, meta),
                    f"{key_prefix}_method_{m}_{sample_idx}",
                    f"factual_{sample_idx}_{m}",
                )
            )
        return out

    st.caption("Graco variants")
    graco_items = _items(GRACO_METHODS)
    if graco_items:
        _show_image_row(graco_items, max_cols=3, width=120)
    else:
        st.caption("No Graco CF tensors found for this sample.")

    st.caption("Comparison baselines")
    base_items = _items(BASELINE_METHODS)
    if base_items:
        _show_image_row(base_items, max_cols=5, width=110)
    else:
        st.caption("No baseline CF tensors found for this sample.")


def _render_population_cf_gallery(
    spec,
    records: list[IndividualRecord],
    orig_alibi: np.ndarray,
    *,
    target_pred_class: int,
    sample_idx: int,
    key_prefix: str,
) -> None:
    candidates = [r for r in records if r.pred_class == target_pred_class]
    st.markdown("**Final-population counterstrokes**")
    if not candidates:
        st.caption("No population individuals for the target class.")
        return
    max_n = len(candidates)
    n_show = st.number_input(
        "Population CF examples to show",
        min_value=1,
        max_value=min(48, max_n),
        value=min(6, max_n),
        step=1,
        key=f"{key_prefix}_n_cf",
        help="How many individuals from the final Graco population (target class) to render.",
    )
    items: list[tuple[np.ndarray, str, str, str]] = []
    for i, rec in enumerate(candidates[: int(n_show)]):
        try:
            cf_disp = alibi_to_display_01(record_to_alibi(spec, rec, orig_alibi))
            items.append(
                (
                    display_to_rgb_uint8(cf_disp),
                    f"pop #{i} · p={rec.pred_prob:.2f}",
                    f"{key_prefix}_pop_{sample_idx}_{i}",
                    f"factual_{sample_idx}_pop{i}",
                )
            )
        except Exception:
            continue
    _show_image_row(items, max_cols=6, width=100)


@st.cache_data(show_spinner=False)
def _cached_landscape(dataset: str, blank_root: str, run_id: int):
    from .datasets import get_app_dataset_spec

    spec = get_app_dataset_spec(dataset)
    return build_belief_landscape(spec, Path(blank_root), run_id=run_id)


@st.cache_data(show_spinner=False)
def _cached_blank_probs(dataset: str, weights: str, device: str):
    import torch
    from .datasets import get_app_dataset_spec

    spec = get_app_dataset_spec(dataset)
    pred, probs, blank = predict_blank(spec, Path(weights), torch.device(device))
    return pred, probs.tolist()


def _tab_factual(cfg: dict) -> None:
    spec = cfg["spec"]
    names = get_class_names(spec.name, str(cfg["class_map"]) if cfg.get("class_map") else None)
    prob_t = float(cfg.get("prob_threshold", 0.5))

    if not cfg["manifest_path"].is_file():
        st.error(f"Manifest not found: {cfg['manifest_path']}")
        return

    entries = _cached_manifest(str(cfg["manifest_path"].resolve()))
    st.subheader("Eval images")
    st.caption(f"{len(entries)} samples — select one, then Analyse.")

    if "factual_selected" not in st.session_state:
        st.session_state.factual_selected = 0

    cols_per_row = 10
    for row_start in range(0, len(entries), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = row_start + j
            if idx >= len(entries):
                break
            ent = entries[idx]
            with col:
                try:
                    _alibi, disp = _cached_factual_image(
                        spec.name,
                        str(cfg["data_root"].resolve()),
                        ent.sample_idx,
                        ent.y_true,
                        ent.split,
                    )
                    thumb = display_to_rgb_uint8(disp)
                    sel = st.session_state.factual_selected == idx
                    if st.button(f"#{idx}", key=f"pick_{idx}", width=button_width(True)):
                        st.session_state.factual_selected = idx
                        st.session_state.factual_run_analyse = False
                        st.rerun()
                    st.image(thumb, width=image_width_pixels(72))
                    st.caption(
                        f"{label_name(names, ent.y_true)} · {ent.sample_idx}"
                        + (" ✓" if sel else "")
                    )
                except Exception:
                    st.caption(f"#{idx} err")

    sel_i = st.session_state.factual_selected
    ent = entries[sel_i]
    st.divider()

    try:
        alibi_preview, disp_preview = _cached_factual_image(
            spec.name,
            str(cfg["data_root"].resolve()),
            ent.sample_idx,
            ent.y_true,
            ent.split,
        )
        pc1, pc2 = st.columns([1, 2])
        with pc1:
            show_downloadable_image(
                display_to_rgb_uint8(disp_preview),
                caption="Selected input",
                width=image_width_pixels(160),
                key=f"factual_input_{ent.sample_idx}",
                filename_stem=f"factual_{ent.sample_idx}_input",
            )
        with pc2:
            st.markdown(
                f"**Manifest #{sel_i}** · sample `{ent.sample_idx}` · "
                f"class **{label_name(names, ent.y_true)}**"
            )
            if st.button("Analyse", type="primary", key="analyse_factual"):
                st.session_state.factual_run_analyse = True
                st.rerun()
    except Exception as ex:
        st.error(str(ex))
        return

    if not st.session_state.get("factual_run_analyse"):
        return

    alibi, disp01 = alibi_preview, disp_preview
    records, src, sample_path = _cached_population(
        str(cfg["factual_graco"].resolve()),
        ent.sample_idx,
        ent.y_true,
        cfg["run_id"],
    )

    if not records:
        st.error(
            f"No final population under `{sample_path or cfg['factual_graco']}` "
            f"for `{ent.sample_idx}_label_{ent.y_true}`."
        )
        return

    with st.expander("Distribution", expanded=True):
        st.info(f"Loaded {len(records)} individuals from **{src}**.")
        result = analyse_factual_distribution(records, prob_threshold=prob_t)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", result.n_total)
        c2.metric("Filtered", result.n_filtered)
        top_name = label_name(names, result.top_class) if result.top_class is not None else "—"
        c3.metric("Top class", top_name)
        c4.metric("Top count", result.top_count)

        if result.distribution:
            st.pyplot(
                fig_class_distribution(
                    result.distribution,
                    names,
                    title=f"pred_class (flip & prob ≥ {prob_t})",
                ),
                clear_figure=True,
            )
        else:
            st.warning("No individuals pass the filter.")
            raw = dict(Counter(r.pred_class for r in records))
            if raw:
                st.pyplot(
                    fig_class_distribution(raw, names, title="Unfiltered"),
                    clear_figure=True,
                )

    if result.top_class is None:
        return

    with st.expander("Change map", expanded=True):
        render_change_map_panel(
            spec=spec,
            orig_alibi=alibi,
            base_display_01=disp01,
            records=result.filtered,
            target_pred_class=result.top_class,
            cfg=cfg,
            title=f"Change map → {top_name}",
            key_prefix="factual_cm",
        )

    with st.expander("Attribution baselines", expanded=True):
        render_attribution_baselines_panel(
            spec=spec,
            sample_idx=ent.sample_idx,
            orig_alibi=alibi,
            base_display_01=disp01,
            target_pred_class=result.top_class,
            cfg=cfg,
            title=f"Attribution baselines → {top_name}",
            key_prefix="factual_attr",
        )

    with st.expander("Counterfactual examples", expanded=True):
        _render_method_cf_gallery(spec, ent.sample_idx, key_prefix="factual_methods")
        st.divider()
        _render_population_cf_gallery(
            spec,
            result.filtered,
            alibi,
            target_pred_class=result.top_class,
            sample_idx=ent.sample_idx,
            key_prefix="factual_pop",
        )

def _render_prob_binned_gallery(
    *,
    cfg: dict,
    spec,
    names: list[str],
    blank: np.ndarray,
    blank_disp: np.ndarray,
    target_class: int,
    records: list[IndividualRecord],
) -> None:
    """One sparsity-first CF per confidence bin + max-SSIM same-class reference."""
    with st.expander("Probability-binned CFs", expanded=True):
        st.caption(
            "One individual per confidence band among predictions of the target class "
            "(sparsity-first: lowest O2, then O3). Each CF is paired with the maximum-SSIM "
            "real image from the same class in the val/test pool."
            + (
                " ImageNet SSIM uses up to 200 images per class."
                if spec.name == "imagenet100"
                else ""
            )
        )
        picks = pick_records_by_prob_bins(records, target_class)
        if not any(p.record is not None for p in picks):
            st.info(
                f"No individuals predict **{label_name(names, target_class)}** "
                "with confidence ≥ 50%."
            )
            return

        graco_mine = load_graco_mine()
        graco_blank = load_graco_blank_evolve()
        try:
            hooks = graco_blank.load_turtle_hooks(spec.graco_dir, dataset=spec.name)
        except Exception as ex:
            st.error(f"Turtle hooks failed: {ex}")
            return

        pool: list[np.ndarray] = []
        data_root = cfg.get("data_root")
        if data_root is not None and Path(data_root).is_dir():
            try:
                cmap = str(cfg["class_map"]) if cfg.get("class_map") else None
                by_class = cached_val_by_class(
                    spec.name,
                    str(Path(data_root).resolve()),
                    cmap,
                )
                pool = list(by_class.get(int(target_class), []))
            except Exception as ex:
                st.warning(f"Val/test pool unavailable for SSIM: {ex}")
        else:
            st.warning("Data root missing — SSIM nearest column disabled.")

        has_weights = Path(cfg["weights"]).is_file() if cfg.get("weights") else False
        import torch

        for pick in picks:
            st.markdown(f"**{pick.label}**")
            if pick.record is None:
                st.caption("No individuals in this band.")
                continue
            rec = pick.record
            try:
                cf_01 = graco_mine.render_cf_01(
                    rec.phenotype,
                    blank,
                    hooks,
                    graco_dir=spec.graco_dir,
                    turtle_delta_scale=spec.turtle_delta_scale,
                )
            except Exception as ex:
                st.warning(f"Render failed ({pick.label}): {ex}")
                continue

            ref_01, ssim = (None, float("nan"))
            if pool:
                try:
                    ref_01, ssim = best_ssim_ref(cf_01, pool, device=cfg["device"])
                except Exception as ex:
                    st.caption(f"SSIM failed ({pick.label}): {ex}")

            c_blank, c_cf, c_ref = st.columns(3)
            base_stem = blank_cf_stem(spec.name, target_class, rec.pred_prob, names)
            widget_key = f"blank_bin_{pick.lo:.1f}_{pick.hi:.1f}_t{target_class}"
            with c_blank:
                show_downloadable_image(
                    display_to_rgb_uint8(blank_disp),
                    caption="Blank",
                    width=image_width_pixels(160),
                    key=f"{widget_key}_blank",
                    filename_stem=f"{dataset_stem(spec.name)}_blank",
                )
            with c_cf:
                cf_alibi = cf_01_to_alibi(spec, cf_01)
                show_downloadable_image(
                    display_to_rgb_uint8(alibi_to_display_01(cf_alibi)),
                    caption=f"CF · p={rec.pred_prob:.3f}",
                    width=image_width_pixels(160),
                    key=f"{widget_key}_cf",
                    filename_stem=base_stem,
                )
                if has_weights:
                    try:
                        dev = torch.device(cfg["device"])
                        st.pyplot(
                            fig_softmax_bars(
                                predict_probs(spec, cf_alibi, cfg["weights"], dev),
                                names,
                                title="CF softmax",
                                highlight_idx=rec.pred_class,
                            ),
                            clear_figure=True,
                        )
                    except Exception:
                        pass
            with c_ref:
                if ref_01 is not None:
                    show_downloadable_image(
                        display_to_rgb_uint8(ref_01),
                        caption=f"SSIM nearest · {ssim:.4f}",
                        width=image_width_pixels(160),
                        key=f"{widget_key}_ssim",
                        filename_stem=blank_ssim_stem(
                            spec.name, target_class, rec.pred_prob, names
                        ),
                    )
                else:
                    st.caption("No SSIM reference")
            st.divider()


def _tab_blank(cfg: dict) -> None:
    spec = cfg["spec"]
    names = get_class_names(spec.name, str(cfg["class_map"]) if cfg.get("class_map") else None)
    prob_t = float(cfg.get("prob_threshold", 0.5))
    blank_root = cfg["blank_graco"]

    st.subheader("Blank canvas belief landscape")
    blank = blank_alibi_for_dataset(spec.name)
    blank_disp = alibi_to_display_01(blank)
    show_downloadable_image(
        display_to_rgb_uint8(blank_disp),
        caption="Starting blank",
        width=image_width_pixels(200),
        key="blank_start",
        filename_stem=f"{dataset_stem(spec.name)}_blank",
    )

    blank_pred: int | None = load_summary_blank_pred(blank_root.parent)
    if cfg["weights"].is_file():
        try:
            blank_pred_live, probs_list = _cached_blank_probs(
                spec.name,
                str(cfg["weights"].resolve()),
                cfg["device"],
            )
            blank_pred = blank_pred_live
            probs = np.array(probs_list, dtype=np.float32)
            st.caption(
                f"Blank prediction: **{label_name(names, blank_pred)}** "
                f"(p={probs[blank_pred]:.3f})"
            )
            st.pyplot(
                fig_softmax_bars(
                    probs,
                    names,
                    title="Baseline softmax (blank)",
                    highlight_idx=blank_pred,
                ),
                clear_figure=True,
            )
        except Exception as ex:
            st.error(f"Classifier on blank failed: {ex}")
    else:
        st.warning("Classifier weights missing — softmax disabled.")

    if blank_pred is None:
        blank_pred = -1

    landscape = _cached_landscape(spec.name, str(blank_root.resolve()), cfg["run_id"])
    available = [r for r in landscape if r.has_csv]
    if not available:
        st.error(f"No blank CSVs under `{blank_root}`.")
        return

    page_size = 25 if spec.num_classes > 20 else spec.num_classes
    page = st.number_input("Landscape page", min_value=0, max_value=max(0, (len(available) - 1) // page_size), value=0)
    chunk = available[page * page_size : (page + 1) * page_size]
    mat = np.array(
        [[r.success_rate, r.mean_p_blank, min(1.0, r.n_flip_hi / max(r.n_total, 1))] for r in chunk],
        dtype=np.float32,
    )
    st.pyplot(
        fig_heatmap_landscape(
            mat,
            [label_name(names, r.wanted_class) for r in chunk],
            ["success@k", "mean_p_blank", "flip_frac"],
            title="Per-target metrics",
        ),
        clear_figure=True,
    )

    if spec.num_classes <= 10:
        grid_cols = st.columns(10)
        for r in landscape:
            if r.wanted_class < 10:
                with grid_cols[r.wanted_class]:
                    st.metric(label_name(names, r.wanted_class), f"{r.success_rate:.0%}")

    target_options = [r.wanted_class for r in landscape if r.has_csv]
    k = st.selectbox(
        "Target class",
        target_options,
        format_func=lambda x: label_name(names, x),
        key="blank_target_sel",
    )
    if blank_pred >= 0 and k == blank_pred:
        st.warning("Target equals blank prediction — weak evolution expected.")

    if st.button("Analyse target", type="primary", key="analyse_blank"):
        st.session_state.blank_run_analyse = True
        st.rerun()

    if not st.session_state.get("blank_run_analyse"):
        return

    drill = load_target_drilldown(blank_root, k, run_id=cfg["run_id"])
    if not drill.records:
        st.error(f"No CSV for target {k}.")
        return

    with st.expander("Population beliefs", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("N total", drill.n_total)
        m2.metric(f"Success @ {label_name(names, k)}", f"{drill.success_rate:.1%}")
        m3.metric("Mean p(blank)", f"{drill.mean_p_blank:.3f}")
        m4.metric("Mode", label_name(names, drill.mode_pred) if drill.mode_pred is not None else "—")

        if drill.dist_full:
            st.pyplot(fig_class_distribution(drill.dist_full, names, title="Full population"), clear_figure=True)
        if drill.dist_filtered:
            st.pyplot(
                fig_class_distribution(
                    drill.dist_filtered,
                    names,
                    title=f"Filtered (≥ {prob_t})",
                ),
                clear_figure=True,
            )
        fig_o1 = fig_o1_histogram(drill.o1_values, title="O1 distribution")
        if fig_o1 is not None:
            st.pyplot(fig_o1, clear_figure=True)

    if cfg["weights"].is_file():
        with st.expander("Best belief (graco50)", expanded=False):
            graco_mine = load_graco_mine()
            graco_blank = load_graco_blank_evolve()
            hooks = graco_blank.load_turtle_hooks(spec.graco_dir, dataset=spec.name)
            rows_mine = [
                graco_mine.Row(
                    o1=r.o1 or 0.0,
                    o2=r.o2 or 0.0,
                    o3=r.o3 or 0.0,
                    original_class=r.original_class,
                    pred_class=r.pred_class,
                    pred_prob=r.pred_prob,
                    phenotype=r.phenotype,
                )
                for r in drill.records
            ]
            row50 = graco_mine.pick_strategies(rows_mine, k).get("graco50")
            if row50 is not None:
                try:
                    import torch

                    cf_01 = graco_mine.render_cf_01(
                        row50.phenotype,
                        blank,
                        hooks,
                        graco_dir=spec.graco_dir,
                        turtle_delta_scale=spec.turtle_delta_scale,
                    )
                    cf_alibi = cf_01_to_alibi(spec, cf_01)
                    dev = torch.device(cfg["device"])
                    c_a, c_b = st.columns(2)
                    with c_a:
                        show_downloadable_image(
                            display_to_rgb_uint8(blank_disp),
                            width=image_width_pixels(200),
                            key="blank_compare_start",
                            filename_stem=f"{dataset_stem(spec.name)}_blank",
                        )
                        st.pyplot(
                            fig_softmax_bars(
                                predict_probs(spec, blank, cfg["weights"], dev),
                                names,
                                title="Blank",
                                highlight_idx=blank_pred if blank_pred >= 0 else None,
                            ),
                            clear_figure=True,
                        )
                    with c_b:
                        show_downloadable_image(
                            display_to_rgb_uint8(alibi_to_display_01(cf_alibi)),
                            width=image_width_pixels(200),
                            key="blank_compare_cf",
                            filename_stem=blank_cf_stem(
                                spec.name, k, float(row50.pred_prob), names
                            ),
                        )
                        st.pyplot(
                            fig_softmax_bars(
                                predict_probs(spec, cf_alibi, cfg["weights"], dev),
                                names,
                                title="CF",
                                highlight_idx=row50.pred_class,
                            ),
                            clear_figure=True,
                        )
                except Exception as ex:
                    st.warning(f"graco50 render: {ex}")

    _render_prob_binned_gallery(
        cfg=cfg,
        spec=spec,
        names=names,
        blank=blank,
        blank_disp=blank_disp,
        target_class=k,
        records=drill.records,
    )

    fact = analyse_factual_distribution(drill.records, prob_threshold=prob_t)
    map_class = k if any(r.pred_class == k for r in fact.filtered) else fact.top_class
    if map_class is not None:
        with st.expander("Change map on blank", expanded=True):
            subset = fact.filtered if fact.filtered else drill.records
            render_change_map_panel(
                spec=spec,
                orig_alibi=blank,
                base_display_01=blank_disp,
                records=subset,
                target_pred_class=map_class,
                cfg=cfg,
                title=f"Blank change map → {label_name(names, map_class)}",
                key_prefix="blank_cm",
                download_stem_fn=lambda method_id, _k=map_class, _n=names: blank_map_stem(
                    spec.name, _k, method_id, _n
                ),
            )


def render_analysis(cfg: dict) -> None:
    cfg = render_toolbar(cfg)
    st.session_state.cfg = cfg
    tab_f, tab_b = st.tabs(["Factual images", "Blank canvas"])
    with tab_f:
        _tab_factual(cfg)
    with tab_b:
        _tab_blank(cfg)
