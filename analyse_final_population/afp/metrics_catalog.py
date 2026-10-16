"""Metric definitions, plot stems, and method ordering for the metrics explorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

METHOD_ORDER = [
    "counterfactual_pt",
    "cfproto",
    "cfrl",
    "leapfactual",
    "leapfactual_r",
    "graco_base_o2",
    "graco_base_o3",
    "graco50",
]

CsvSource = Literal[
    "per_sample",
    "distribution",
    "batch_fid",
    "cmmd_per_sample",
    "cmmd_batch",
    "summary",
]

Direction = Literal["lower", "higher", "context"]


@dataclass(frozen=True)
class MetricDef:
    id: str
    label: str
    group: str
    description: str
    how_computed: str
    formula: str | None
    direction: Direction
    csv_source: CsvSource
    csv_column: str | None
    plot_stems: tuple[str, ...]
    dataset_notes: dict[str, str] | None = None


METRIC_GROUPS: list[str] = [
    "Success",
    "Proximity",
    "Classifier",
    "Embedding",
    "Perceptual",
    "Distribution",
    "Batch / stratified",
]

DEFAULT_METRIC_IDS = [
    "success_class_changed_rate",
    "l2_distance",
    "frechet_inception_distance_cf_vs_originals",
    "cmmd_orig_vs_cf_distribution",
]

METRICS: list[MetricDef] = [
    MetricDef(
        id="success_class_changed_rate",
        label="Success rate (class changed)",
        group="Success",
        description="Fraction of eval samples where the counterfactual achieves a successful flip.",
        how_computed=(
            "Baselines: pred_class_cf ≠ y_true on saved CF tensors. "
            "Graco methods: selected-row o1 < 0 from Graco CSV in meta.json."
        ),
        formula=None,
        direction="higher",
        csv_source="per_sample",
        csv_column="pred_class_cf",
        plot_stems=("01_success_rate_class_changed",),
    ),
    MetricDef(
        id="l2_distance",
        label="L2 distance (proximity)",
        group="Proximity",
        description="Euclidean distance between original and counterfactual in pixel space.",
        how_computed="Computed on orig_nhwc and cf_* tensors in Alibi range [-0.5, 0.5].",
        formula=r"\|x_{\mathrm{cf}} - x_{\mathrm{orig}}\|_2",
        direction="lower",
        csv_source="per_sample",
        csv_column="l2_distance",
        plot_stems=("02_per_sample_means_grid_lower", "03_per_sample_boxplots_lower"),
    ),
    MetricDef(
        id="sparsity_count",
        label="Sparsity count",
        group="Proximity",
        description="Number of pixels changed beyond a small threshold.",
        how_computed="Count pixels where |x_cf − x_orig| > ε with ε = 1/255.",
        formula=r"\#\{p : |x_{\mathrm{cf}}(p) - x_{\mathrm{orig}}(p)| > \epsilon\}",
        direction="lower",
        csv_source="per_sample",
        csv_column="sparsity_count",
        plot_stems=("02_per_sample_means_grid_lower", "03_per_sample_boxplots_lower"),
    ),
    MetricDef(
        id="sparsity_frac",
        label="Sparsity fraction",
        group="Proximity",
        description="Fraction of pixels changed beyond ε = 1/255.",
        how_computed="sparsity_count divided by total pixels.",
        formula=r"\frac{\mathrm{sparsity\_count}}{H \times W \times C}",
        direction="lower",
        csv_source="per_sample",
        csv_column="sparsity_frac",
        plot_stems=("02_per_sample_means_grid_lower", "03_per_sample_boxplots_lower"),
    ),
    MetricDef(
        id="p_true_orig",
        label="P(true class) on original",
        group="Classifier",
        description="Task classifier probability of the true label on the original image.",
        how_computed="orig_nhwc passed through the dataset classifier (MNISTNet / CifarResNet / Inception).",
        formula=r"p(y_{\mathrm{true}} \mid x_{\mathrm{orig}})",
        direction="context",
        csv_source="per_sample",
        csv_column="p_true_orig",
        plot_stems=("04_probabilities_panel_means", "03_per_sample_boxplots_probabilities"),
    ),
    MetricDef(
        id="p_true_cf",
        label="P(true class) on CF",
        group="Classifier",
        description="Task classifier probability of the true label on the counterfactual.",
        how_computed="cf_* passed through the dataset classifier.",
        formula=r"p(y_{\mathrm{true}} \mid x_{\mathrm{cf}})",
        direction="context",
        csv_source="per_sample",
        csv_column="p_true_cf",
        plot_stems=("04_probabilities_panel_means", "03_per_sample_boxplots_probabilities"),
    ),
    MetricDef(
        id="delta_p_true",
        label="Δ P(true class)",
        group="Classifier",
        description="Change in true-class probability after the counterfactual edit.",
        how_computed="p_true_cf − p_true_orig.",
        formula=r"\Delta p = p_{\mathrm{true, cf}} - p_{\mathrm{true, orig}}",
        direction="context",
        csv_source="per_sample",
        csv_column="delta_p_true",
        plot_stems=("04_probabilities_panel_means", "03_per_sample_boxplots_probabilities"),
    ),
    MetricDef(
        id="p_intended_target_cf",
        label="P(intended target) on CF",
        group="Classifier",
        description="Classifier probability of the intended target class on the CF (targeted methods).",
        how_computed="Uses y_intended_target from meta.json / manifest.",
        formula=r"p(y_{\mathrm{target}} \mid x_{\mathrm{cf}})",
        direction="context",
        csv_source="per_sample",
        csv_column="p_intended_target_cf",
        plot_stems=("04_probabilities_panel_means", "03_per_sample_boxplots_probabilities"),
    ),
    MetricDef(
        id="p_pred_cf",
        label="P(predicted class) on CF",
        group="Classifier",
        description="Probability of the argmax predicted class on the counterfactual.",
        how_computed="pred_class_cf = argmax; p_pred_cf is the corresponding softmax value.",
        formula=r"p(\hat{y}_{\mathrm{cf}} \mid x_{\mathrm{cf}})",
        direction="context",
        csv_source="per_sample",
        csv_column="p_pred_cf",
        plot_stems=("04_probabilities_panel_means", "03_per_sample_boxplots_probabilities"),
    ),
    MetricDef(
        id="inception_sqdist_to_source_class_mean",
        label="Embedding dist. to source class mean",
        group="Embedding",
        description="Squared L2 distance from CF embedding to the mean embedding of real images in the true class.",
        how_computed=(
            "Not FID. Per-class mean embedding over a reference pool; distance computed per CF."
        ),
        formula=r"\| \phi(x_{\mathrm{cf}}) - \mu_{y_{\mathrm{true}}} \|_2^2",
        direction="lower",
        csv_source="per_sample",
        csv_column="inception_sqdist_to_source_class_mean",
        plot_stems=("02_per_sample_means_grid_lower", "03_per_sample_boxplots_lower"),
        dataset_notes={
            "mnist": "Uses Inception v3 pooled features; reference pool = train indices ≥ 100.",
            "cifar10": "Column name says inception_* but features are ResNet-18 trunk (eval script convention).",
            "imagenet100": "Uses Inception v3 pooled features; pool from train_pool_index.json.",
        },
    ),
    MetricDef(
        id="inception_sqdist_to_predicted_class_mean",
        label="Embedding dist. to predicted class mean",
        group="Embedding",
        description="Squared L2 distance from CF embedding to mean embedding of the predicted CF class.",
        how_computed="Same embedding network as source-class column; mean taken over predicted class pool.",
        formula=r"\| \phi(x_{\mathrm{cf}}) - \mu_{\hat{y}_{\mathrm{cf}}} \|_2^2",
        direction="lower",
        csv_source="per_sample",
        csv_column="inception_sqdist_to_predicted_class_mean",
        plot_stems=("02_per_sample_means_grid_lower", "03_per_sample_boxplots_lower"),
        dataset_notes={
            "cifar10": "ResNet-18 trunk features despite inception_* column name.",
        },
    ),
    MetricDef(
        id="lpips_to_original",
        label="LPIPS to original",
        group="Perceptual",
        description="Learned perceptual similarity vs the original image (lower = closer).",
        how_computed="LPIPS network; MNIST/CIFAR upsampled to 64×64×3, ImageNet to 256×256.",
        formula=r"\mathrm{LPIPS}(x_{\mathrm{cf}}, x_{\mathrm{orig}})",
        direction="lower",
        csv_source="per_sample",
        csv_column="lpips_to_original",
        plot_stems=("02_per_sample_means_grid_lower", "03_per_sample_boxplots_lower"),
    ),
    MetricDef(
        id="lpips_to_random_pred_class",
        label="LPIPS to random same-class ref",
        group="Perceptual",
        description="LPIPS vs a random real image from the predicted class reference pool.",
        how_computed="Reference image sampled from real-image pool of pred_class_cf.",
        formula=r"\mathrm{LPIPS}(x_{\mathrm{cf}}, x_{\mathrm{ref}})",
        direction="lower",
        csv_source="per_sample",
        csv_column="lpips_to_random_pred_class",
        plot_stems=("02_per_sample_means_grid_lower", "03_per_sample_boxplots_lower"),
    ),
    MetricDef(
        id="ssim_to_original",
        label="SSIM to original",
        group="Perceptual",
        description="Structural similarity vs original (higher = closer).",
        how_computed="SSIM between CF and original in display space.",
        formula=r"\mathrm{SSIM}(x_{\mathrm{cf}}, x_{\mathrm{orig}})",
        direction="higher",
        csv_source="per_sample",
        csv_column="ssim_to_original",
        plot_stems=("02_per_sample_means_grid_higher", "03_per_sample_boxplots_higher"),
    ),
    MetricDef(
        id="ssim_to_random_pred_class",
        label="SSIM to random same-class ref",
        group="Perceptual",
        description="SSIM vs random reference from predicted class pool (higher = closer).",
        how_computed="Reference from real-image pool of pred_class_cf.",
        formula=r"\mathrm{SSIM}(x_{\mathrm{cf}}, x_{\mathrm{ref}})",
        direction="higher",
        csv_source="per_sample",
        csv_column="ssim_to_random_pred_class",
        plot_stems=("02_per_sample_means_grid_higher", "03_per_sample_boxplots_higher"),
    ),
    MetricDef(
        id="frechet_inception_distance_cf_vs_originals",
        label="FID (paired orig vs CF)",
        group="Distribution",
        description="Fréchet Inception Distance between paired originals and counterfactuals.",
        how_computed="torch-fidelity on rendered PNG folders per method (valid CFs only).",
        formula=r"\mathrm{FID}(\{x_{\mathrm{orig}}\}, \{x_{\mathrm{cf}}\})",
        direction="lower",
        csv_source="distribution",
        csv_column="frechet_inception_distance_cf_vs_originals",
        plot_stems=("10_distribution_frechet_paired_orig_cf", "14_distribution_metrics_heatmap"),
    ),
    MetricDef(
        id="kid_mean_cf_vs_originals",
        label="KID mean (paired orig vs CF)",
        group="Distribution",
        description="Kernel Inception Distance mean between originals and CFs.",
        how_computed="torchmetrics KernelInceptionDistance on paired sets.",
        formula=r"\mathrm{KID}_{\mathrm{mean}}",
        direction="lower",
        csv_source="distribution",
        csv_column="kid_mean_cf_vs_originals",
        plot_stems=("11_distribution_kid_mean", "14_distribution_metrics_heatmap"),
    ),
    MetricDef(
        id="kid_std_cf_vs_originals",
        label="KID std (paired orig vs CF)",
        group="Distribution",
        description="Standard deviation of KID estimate.",
        how_computed="torchmetrics KID std over bootstrap / estimator.",
        formula=r"\mathrm{KID}_{\mathrm{std}}",
        direction="lower",
        csv_source="distribution",
        csv_column="kid_std_cf_vs_originals",
        plot_stems=("12_distribution_kid_std", "14_distribution_metrics_heatmap"),
    ),
    MetricDef(
        id="cmmd_orig_vs_cf_distribution",
        label="CLIP CMMD (orig vs CF distribution)",
        group="Distribution",
        description="Gaussian-kernel MMD between CLIP embeddings of originals and CFs.",
        how_computed="OpenCLIP ViT-L/14@336 embeddings; MMD between the two sets per method.",
        formula=r"\mathrm{MMD}^2_{\mathrm{CLIP}}(\{x_{\mathrm{orig}}\}, \{x_{\mathrm{cf}}\})",
        direction="lower",
        csv_source="distribution",
        csv_column="cmmd_orig_vs_cf_distribution",
        plot_stems=("13_distribution_cmmd_paired", "14_distribution_metrics_heatmap"),
    ),
    MetricDef(
        id="frechet_inception_distance_vs_source_y_true",
        label="Batch FID vs source class pool",
        group="Batch / stratified",
        description="FID between CF set and stratified real images (true-class stratification).",
        how_computed="torch-fidelity batch comparison in batch_fid.csv.",
        formula=r"\mathrm{FID}_{\mathrm{stratified, source}}",
        direction="lower",
        csv_source="batch_fid",
        csv_column="frechet_inception_distance_vs_source_y_true",
        plot_stems=("20_batch_fid_stratified",),
    ),
    MetricDef(
        id="frechet_inception_distance_vs_predicted_class",
        label="Batch FID vs predicted class pool",
        group="Batch / stratified",
        description="FID between CF set and stratified real images (predicted-class stratification).",
        how_computed="torch-fidelity batch comparison in batch_fid.csv.",
        formula=r"\mathrm{FID}_{\mathrm{stratified, pred}}",
        direction="lower",
        csv_source="batch_fid",
        csv_column="frechet_inception_distance_vs_predicted_class",
        plot_stems=("20_batch_fid_stratified",),
    ),
    MetricDef(
        id="cmmd_vs_source_y_true_class",
        label="Per-sample CMMD vs source class",
        group="Batch / stratified",
        description="Per-CF CLIP MMD vs reference pool of the true class.",
        how_computed="cmmd_per_sample.csv — one row per (sample, method).",
        formula=r"\mathrm{MMD}_{\mathrm{CLIP}}(\phi(x_{\mathrm{cf}}), \mathcal{R}_{y_{\mathrm{true}}})",
        direction="lower",
        csv_source="cmmd_per_sample",
        csv_column="cmmd_vs_source_y_true_class",
        plot_stems=("22_cmmd_per_sample_means",),
    ),
    MetricDef(
        id="cmmd_vs_predicted_class",
        label="Per-sample CMMD vs predicted class",
        group="Batch / stratified",
        description="Per-CF CLIP MMD vs reference pool of the predicted CF class.",
        how_computed="cmmd_per_sample.csv.",
        formula=r"\mathrm{MMD}_{\mathrm{CLIP}}(\phi(x_{\mathrm{cf}}), \mathcal{R}_{\hat{y}_{\mathrm{cf}}})",
        direction="lower",
        csv_source="cmmd_per_sample",
        csv_column="cmmd_vs_predicted_class",
        plot_stems=("22_cmmd_per_sample_means",),
    ),
    MetricDef(
        id="cmmd_vs_stratified_source_y_true",
        label="Stratified batch CMMD (source)",
        group="Batch / stratified",
        description="MMD between CF set and stratified real-image pool (true class).",
        how_computed="cmmd_batch_stratified.csv.",
        formula=None,
        direction="lower",
        csv_source="cmmd_batch",
        csv_column="cmmd_vs_stratified_source_y_true",
        plot_stems=("21_cmmd_batch_stratified",),
    ),
    MetricDef(
        id="cmmd_vs_stratified_predicted_class",
        label="Stratified batch CMMD (predicted)",
        group="Batch / stratified",
        description="MMD between CF set and stratified real-image pool (predicted class).",
        how_computed="cmmd_batch_stratified.csv.",
        formula=None,
        direction="lower",
        csv_source="cmmd_batch",
        csv_column="cmmd_vs_stratified_predicted_class",
        plot_stems=("21_cmmd_batch_stratified",),
    ),
]

METRICS_BY_ID: dict[str, MetricDef] = {m.id: m for m in METRICS}


def metrics_for_group(group: str) -> list[MetricDef]:
    return [m for m in METRICS if m.group == group]


def ordered_methods_present(methods: set[str] | list[str]) -> list[str]:
    present = set(methods)
    out = [m for m in METHOD_ORDER if m in present]
    out.extend(sorted(present - set(out)))
    return out
