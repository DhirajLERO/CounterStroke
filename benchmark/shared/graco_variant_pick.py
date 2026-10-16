"""Pick Graco benchmark variants (o2, o3, graco50) from evolution candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from evaluation.metrics.effectiveness import effectiveness


@dataclass
class EvolvedCandidate:
    phenotype: str
    o1: float
    o2: float
    o3: float
    image_np: np.ndarray | None = None


GRACO_VARIANT_NAMES = ("graco_base_o2", "graco_base_o3", "graco50")


def _predictor_device(predictors: dict) -> torch.device:
    for clf in predictors.values():
        return next(clf.parameters()).device
    return torch.device("cpu")


def _batch_on_predictor_device(batch: dict, predictors: dict) -> dict:
    device = _predictor_device(predictors)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _pick_min_o2_then_o3(cands: list[EvolvedCandidate]) -> EvolvedCandidate | None:
    if not cands:
        return None
    return min(cands, key=lambda r: (r.o2, r.o3, r.o1))


def _pick_min_o3_then_o2(cands: list[EvolvedCandidate]) -> EvolvedCandidate | None:
    if not cands:
        return None
    return min(cands, key=lambda r: (r.o3, r.o2, r.o1))


def _intervention_success_morphomnist(
    cf_batch: dict,
    do_parent: str,
    targets: dict,
    predictors: dict,
    unnormalize_fn,
) -> bool:
    eff = effectiveness(cf_batch, unnormalize_fn, predictors, "morphomnist")
    if do_parent == "digit":
        return float(eff.get("digit", 0.0)) >= 0.99
    if do_parent in eff:
        # thickness/intensity: effectiveness returns low MAE — treat as success if score < 0.15
        return float(eff[do_parent]) < 0.15
    return False


def _intervention_success_celeba(
    cf_batch: dict,
    do_parent: str,
    targets: dict,
    predictors: dict,
) -> bool:
    eff = effectiveness(cf_batch, unnormalize_fn=None, predictors=predictors, dataset="celeba")
    return float(eff.get(do_parent, 0.0)) >= 0.5


def _strong_success_morphomnist(
    cf_batch: dict,
    do_parent: str,
    targets: dict,
    predictors: dict,
    unnormalize_fn,
) -> bool:
    if do_parent == "digit":
        return _intervention_success_morphomnist(cf_batch, do_parent, targets, predictors, unnormalize_fn)
    xi = cf_batch["image"]
    tgt = cf_batch[do_parent]
    if do_parent == "thickness":
        pred = predictors["thickness"](xi, cf_batch.get("intensity"))
        err = float(nn.MSELoss()(pred, tgt.float().view_as(pred)).item())
        return err < 0.05
    pred = torch.sigmoid(predictors[do_parent](xi))
    err = float(nn.MSELoss()(pred, tgt.float().view_as(pred)).item())
    return err < 0.05


def _strong_success_celeba(cf_batch: dict, do_parent: str, targets: dict, predictors: dict) -> bool:
    xi = cf_batch["image"]
    tgt = cf_batch[do_parent]
    clf = predictors[do_parent]
    if getattr(clf, "conditions", None) and clf.conditions.get(do_parent):
        cond = torch.cat([cf_batch[att] for att in clf.conditions[do_parent]], dim=1)
        pred = torch.sigmoid(clf(xi, cond))
    else:
        pred = torch.sigmoid(clf(xi))
    return float((pred >= 0.5).float().eq(tgt.float().view_as(pred)).float().mean().item()) >= 0.99


def filter_successful_candidates(
    candidates: list[EvolvedCandidate],
    *,
    do_parent: str,
    targets: dict,
    predictors: dict,
    dataset_name: str,
    unnormalize_fn,
    image_from_phenotype,
) -> list[EvolvedCandidate]:
    """Keep candidates whose rendered image achieves the intervention on do_parent."""
    ok: list[EvolvedCandidate] = []
    for c in candidates:
        if c.image_np is None:
            continue
        cf_tensor = image_from_phenotype(c.image_np)
        cf_batch = _batch_on_predictor_device({**targets, "image": cf_tensor}, predictors)
        if dataset_name == "morphomnist":
            success = _intervention_success_morphomnist(cf_batch, do_parent, targets, predictors, unnormalize_fn)
        else:
            success = _intervention_success_celeba(cf_batch, do_parent, targets, predictors)
        if success:
            ok.append(c)
    return ok


def filter_graco50_candidates(
    candidates: list[EvolvedCandidate],
    *,
    do_parent: str,
    targets: dict,
    predictors: dict,
    dataset_name: str,
    unnormalize_fn,
    image_from_phenotype,
) -> list[EvolvedCandidate]:
    ok: list[EvolvedCandidate] = []
    for c in candidates:
        if c.image_np is None:
            continue
        cf_tensor = image_from_phenotype(c.image_np)
        cf_batch = _batch_on_predictor_device({**targets, "image": cf_tensor}, predictors)
        if dataset_name == "morphomnist":
            strong = _strong_success_morphomnist(cf_batch, do_parent, targets, predictors, unnormalize_fn)
        else:
            strong = _strong_success_celeba(cf_batch, do_parent, targets, predictors)
        if strong:
            ok.append(c)
    return ok


def pick_benchmark_variants(
    candidates: list[EvolvedCandidate],
    *,
    do_parent: str,
    targets: dict,
    predictors: dict,
    dataset_name: str,
    unnormalize_fn,
    image_from_phenotype,
) -> dict[str, EvolvedCandidate | None]:
    """Return up to three variant picks; fall back to min-o1 if no successful candidate."""
    successful = filter_successful_candidates(
        candidates,
        do_parent=do_parent,
        targets=targets,
        predictors=predictors,
        dataset_name=dataset_name,
        unnormalize_fn=unnormalize_fn,
        image_from_phenotype=image_from_phenotype,
    )
    pool = successful if successful else candidates
    graco50_pool = filter_graco50_candidates(
        candidates,
        do_parent=do_parent,
        targets=targets,
        predictors=predictors,
        dataset_name=dataset_name,
        unnormalize_fn=unnormalize_fn,
        image_from_phenotype=image_from_phenotype,
    )
    fallback = min(candidates, key=lambda c: c.o1) if candidates else None
    return {
        "graco_base_o2": _pick_min_o2_then_o3(pool) or fallback,
        "graco_base_o3": _pick_min_o3_then_o2(pool) or fallback,
        "graco50": _pick_min_o2_then_o3(graco50_pool if graco50_pool else pool) or fallback,
    }
