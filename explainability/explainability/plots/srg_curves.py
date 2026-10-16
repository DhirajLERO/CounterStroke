"""Per-method MIF/LIF curve plots for SRG evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..srg.evaluate import SRGResult


def save_mif_lif_plot(
    path: Path,
    result: SRGResult,
    *,
    method: str,
    prob_class: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(result.k_fractions, result.mif_curve, label="MIF", color="C0", linewidth=2)
    ax.plot(result.k_fractions, result.lif_curve, label="LIF", color="C1", linewidth=2)
    ax.axhline(
        float(result.p_baseline),
        color="0.45",
        linestyle="--",
        linewidth=1,
        label=f"p0 (class {prob_class})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fraction of pixels corrupted")
    ax.set_ylabel(f"p(class {prob_class})")
    ax.set_title(f"{method}\nSRG={result.srg_score:.4f}  occlusion={result.occlusion_strategy}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_mif_lif_grid(
    path: Path,
    curves: dict[str, SRGResult],
    *,
    prob_classes: dict[str, int],
) -> None:
    if not curves:
        return
    methods = sorted(curves.keys())
    n = len(methods)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)

    for i, method in enumerate(methods):
        ax = axes[i // cols][i % cols]
        result = curves[method]
        cls = prob_classes[method]
        ax.plot(result.k_fractions, result.mif_curve, label="MIF", color="C0", linewidth=1.5)
        ax.plot(result.k_fractions, result.lif_curve, label="LIF", color="C1", linewidth=1.5)
        ax.axhline(float(result.p_baseline), color="0.45", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{method}\nSRG={result.srg_score:.3f}  p(c{cls})", fontsize=9)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("MIF / LIF curves (all methods)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
