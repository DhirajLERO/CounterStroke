"""Orchestrate blank-canvas Graco runs + PDF for one dataset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .blank_canvas import blank_cifar_hwc, blank_imagenet_chw, blank_mnist_hw
from .classifier_load import load_cifar_classifier, load_imagenet_classifier, load_mnist_classifier
from .graco_blank_evolve import evolve_blank_canvas, load_turtle_hooks
from .graco_mine import pick_strategies, read_final_generation_csv, render_cf_01
from .paths import ANALYSIS_ROOT, CIFAR_DIR, IMNET_DIR, MNIST_DIR
from .pdf_report import write_blank_graco_pdf
from .ssim import best_ssim_match
from .val_by_class import (
    class_names_cifar,
    class_names_imagenet,
    class_names_mnist,
    val_pool_cifar,
    val_pool_imagenet,
    val_pool_mnist,
)


@dataclass
class DatasetSpec:
    name: str
    num_classes: int
    graco_dir: Path
    default_data_root: Path
    default_weights: Path
    blank_fn: Any
    load_classifier: Any
    val_pool_fn: Any
    class_names_fn: Any
    predict_blank: Any


def _predict_mnist(model, blank_hw: np.ndarray, device: torch.device) -> int:
    sys.path.insert(0, str(MNIST_DIR / "graco"))
    from infer_utils import predict_probs_numpy  # noqa: WPS433

    x = blank_hw[np.newaxis, ...]
    pred_cls, _, _ = predict_probs_numpy(model, x, device=device)
    return int(pred_cls[0])


def _predict_cifar(model, blank_hwc: np.ndarray, device: torch.device) -> int:
    sys.path.insert(0, str(CIFAR_DIR / "graco"))
    from infer_utils import predict_probs_numpy  # noqa: WPS433

    x = blank_hwc[np.newaxis, ...]
    pred_cls, _, _ = predict_probs_numpy(model, x, device=device)
    return int(pred_cls[0])


def _predict_imagenet(model, blank_chw: np.ndarray, device: torch.device) -> int:
    sys.path.insert(0, str(IMNET_DIR / "graco"))
    from inception_infer import predict_probs_numpy_inception  # noqa: WPS433

    x = blank_chw[np.newaxis, ...]
    pred_cls, _, _ = predict_probs_numpy_inception(model, x, device=device)
    return int(pred_cls[0])


def get_dataset_spec(name: str) -> DatasetSpec:
    if name == "mnist":
        return DatasetSpec(
            name="mnist",
            num_classes=10,
            graco_dir=MNIST_DIR / "graco",
            default_data_root=MNIST_DIR / "data",
            default_weights=MNIST_DIR / "models" / "mnist_cnn.pt",
            blank_fn=blank_mnist_hw,
            load_classifier=load_mnist_classifier,
            val_pool_fn=val_pool_mnist,
            class_names_fn=lambda: class_names_mnist(),
            predict_blank=_predict_mnist,
        )
    if name == "cifar10":
        return DatasetSpec(
            name="cifar10",
            num_classes=10,
            graco_dir=CIFAR_DIR / "graco",
            default_data_root=CIFAR_DIR / "data",
            default_weights=CIFAR_DIR / "models" / "cifar_resnet.pt",
            blank_fn=blank_cifar_hwc,
            load_classifier=load_cifar_classifier,
            val_pool_fn=val_pool_cifar,
            class_names_fn=class_names_cifar,
            predict_blank=_predict_cifar,
        )
    if name == "imagenet100":
        return DatasetSpec(
            name="imagenet100",
            num_classes=100,
            graco_dir=IMNET_DIR / "graco",
            default_data_root=IMNET_DIR / "data" / "imagenet100",
            default_weights=IMNET_DIR / "models" / "imagenet100_inception.pt",
            blank_fn=blank_imagenet_chw,
            load_classifier=load_imagenet_classifier,
            val_pool_fn=val_pool_imagenet,
            class_names_fn=class_names_imagenet,
            predict_blank=_predict_imagenet,
        )
    raise ValueError(f"Unknown dataset {name!r}")


def _run_one_class(
    wanted_class: int,
    spec: DatasetSpec,
    blank: np.ndarray,
    label: int,
    model,
    hooks,
    cfg: dict,
    graco_root: Path,
    val_by_class: dict[int, list[np.ndarray]],
    turtle_delta_scale: float,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    results_dir = graco_root / f"target_{wanted_class:03d}" / "run_0"
    csv_path = results_dir / "run_0_final_generation.csv"
    if not cfg.get("_skip_graco"):
        csv_path = evolve_blank_canvas(
            hooks=hooks,
            model=model,
            blank_image=blank,
            label=label,
            wanted_class=wanted_class,
            cfg={**cfg, "_graco_dir": str(spec.graco_dir), "device": str(device)},
            results_dir=results_dir,
            device=device,
        )

    if not csv_path.is_file():
        return wanted_class, {}

    rows = read_final_generation_csv(csv_path)
    picked = pick_strategies(rows, wanted_class)
    page: dict[str, Any] = {}
    pool = val_by_class.get(wanted_class, [])

    for variant, row in picked.items():
        if row is None:
            page[variant] = None
            continue
        cf_01 = render_cf_01(
            row.phenotype,
            blank,
            hooks,
            graco_dir=spec.graco_dir,
            turtle_delta_scale=turtle_delta_scale,
        )
        ref_i, ssim = best_ssim_match(cf_01, pool, device=device)
        ref_01 = pool[ref_i] if ref_i >= 0 else None
        page[variant] = {
            "cf_01": cf_01,
            "ref_01": ref_01,
            "pred_prob": row.pred_prob,
            "ssim": ssim,
            "pred_class": row.pred_class,
        }
    return wanted_class, page


def run_blank_graco_analysis(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Blank-canvas Graco class analysis + PDF")
    p.add_argument("--dataset", required=True, choices=["mnist", "cifar10", "imagenet100"])
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--imagenet-root", type=Path, default=None, help="Alias for --data-root (ImageNet-100)")
    p.add_argument("--classifier-weights", type=Path, default=None)
    p.add_argument("--class-map", type=Path, default=None, help="ImageNet-100 only")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--test-run", action="store_true")
    p.add_argument("--skip-graco", action="store_true", help="PDF only from existing CSVs")
    p.add_argument("--classes", type=str, default=None, help="e.g. 0,1,2")
    p.add_argument("--parallel", type=int, default=1, help="Parallel class jobs (separate processes)")
    args = p.parse_args(argv)

    spec = get_dataset_spec(args.dataset)
    out_dir = args.out_dir or (ANALYSIS_ROOT / spec.name / "outputs")
    out_dir = out_dir.resolve()
    graco_root = out_dir / "graco_run"
    graco_root.mkdir(parents=True, exist_ok=True)

    cfg_path = args.config or (ANALYSIS_ROOT / spec.name / "config_blank.json")
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    if args.test_run:
        cfg["population_size"] = 2000
        cfg["max_generations"] = 500
        cfg["elite_size"] = 1
        cfg["halloffame_size"] = 3

    data_root = (args.data_root or args.imagenet_root or spec.default_data_root).resolve()
    weights = (args.classifier_weights or spec.default_weights).resolve()
    if not weights.is_file():
        raise SystemExit(f"Missing classifier weights: {weights}")

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    model = spec.load_classifier(weights, device)

    blank = spec.blank_fn()
    label = spec.predict_blank(model, blank, device)
    hooks = load_turtle_hooks(spec.graco_dir, dataset=spec.name)

    if spec.name == "imagenet100":
        val_by_class = spec.val_pool_fn(data_root, args.class_map)
        names = spec.class_names_fn(args.class_map)
    else:
        val_by_class = spec.val_pool_fn(data_root)
        names = spec.class_names_fn()

    if blank.ndim == 3 and blank.shape[0] == 3:
        sys.path.insert(0, str(spec.graco_dir))
        from apply_turtle_rgb import alibi_chw_to_hwc_01  # noqa: WPS433

        blank_display = alibi_chw_to_hwc_01(blank)
    else:
        blank_display = hooks.to_display_01(blank)

    if args.classes:
        classes = [int(x.strip()) for x in args.classes.split(",") if x.strip()]
    else:
        classes = list(range(spec.num_classes))

    cfg_run = {**cfg, "_skip_graco": args.skip_graco}
    turtle_delta_scale = float(cfg.get("turtle_delta_scale", 5.0))
    pages: dict[int, dict[str, Any]] = {}

    if args.parallel <= 1:
        for k in classes:
            print(f"[{spec.name}] target class {k}/{spec.num_classes - 1} ...")
            _, page = _run_one_class(
                k,
                spec,
                blank,
                label,
                model,
                hooks,
                cfg_run,
                graco_root,
                val_by_class,
                turtle_delta_scale,
                device,
            )
            pages[k] = page
    else:
        raise NotImplementedError("Use --parallel 1 for now (GPU-bound sequential runs)")

    pdf_path = out_dir / "blank_class_graco_report.pdf"
    write_blank_graco_pdf(
        out_pdf=pdf_path,
        class_names=names,
        blank_pred=label,
        blank_display_01=blank_display,
        pages=pages,
    )

    summary = {
        "dataset": spec.name,
        "blank_pred": label,
        "data_root": str(data_root),
        "classes": classes,
        "pdf": str(pdf_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {pdf_path}")
