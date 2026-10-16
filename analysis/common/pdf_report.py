"""Multi-page PDF: per-class Graco variants + SSIM-matched val references."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import numpy as np  # noqa: E402

VARIANT_ORDER = (
    "graco_base_o2",
    "graco_base_o3",
    "graco50",
    "graco60",
    "graco70",
    "graco80",
    "graco90",
)
VARIANT_LABELS = {
    "graco_base_o2": "Graco base O2",
    "graco_base_o3": "Graco base O3",
    "graco50": "Graco50",
    "graco60": "Graco60",
    "graco70": "Graco70",
    "graco80": "Graco80",
    "graco90": "Graco90",
}


def _show(ax, img: np.ndarray, title: str) -> None:
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 2:
        ax.imshow(a, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(np.clip(a, 0.0, 1.0))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def write_blank_graco_pdf(
    *,
    out_pdf: Path,
    class_names: list[str],
    blank_pred: int,
    blank_display_01: np.ndarray,
    pages: dict[int, dict[str, Any]],
) -> None:
    """
  ``pages[k]`` keys: variant name -> dict with cf_01, ref_01, pred_prob, ssim, row meta.
    """
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for k in sorted(pages.keys()):
            name = class_names[k] if k < len(class_names) else str(k)
            fig = plt.figure(figsize=(12, 10), dpi=120)
            fig.suptitle(
                f"Class {k}: {name}  |  blank_pred={blank_pred}  |  target={k}",
                fontsize=12,
                y=0.98,
            )
            n_rows = 1 + len(VARIANT_ORDER)
            height_ratios = [0.8] + [1.0] * len(VARIANT_ORDER)
            gs = fig.add_gridspec(
                n_rows, 2, height_ratios=height_ratios, hspace=0.35, wspace=0.15
            )

            ax_blank = fig.add_subplot(gs[0, :])
            _show(ax_blank, blank_display_01, "Blank start")
            ax_blank.text(
                0.5,
                -0.08,
                f"Classifier on blank: class {blank_pred}",
                transform=ax_blank.transAxes,
                ha="center",
                fontsize=9,
            )

            for row_i, variant in enumerate(VARIANT_ORDER, start=1):
                info = pages[k].get(variant) or {}
                cf = info.get("cf_01")
                ref = info.get("ref_01")
                ax_l = fig.add_subplot(gs[row_i, 0])
                ax_r = fig.add_subplot(gs[row_i, 1])
                if cf is None:
                    _show(ax_l, np.zeros((8, 8)), f"{VARIANT_LABELS[variant]} (none)")
                else:
                    cap = VARIANT_LABELS[variant]
                    if info.get("pred_prob") is not None:
                        cap += f"  p={info['pred_prob']:.3f}"
                    _show(ax_l, cf, cap)
                if ref is None:
                    _show(ax_r, np.zeros((8, 8)), "Best SSIM val (none)")
                else:
                    ssim = info.get("ssim", float("nan"))
                    _show(ax_r, ref, f"Val match  SSIM={ssim:.4f}")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
