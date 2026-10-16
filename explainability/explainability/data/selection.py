"""Phenotype filtering for explanation class."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .population import IndividualRecord

PROB_THRESHOLD = 0.5


def filter_flipped_high_conf(
    records: list[IndividualRecord],
    *,
    prob_threshold: float = PROB_THRESHOLD,
    original_class: int | None = None,
) -> list[IndividualRecord]:
    oc = original_class
    return [
        r
        for r in records
        if (oc if oc is not None and oc >= 0 else r.original_class) != r.pred_class
        and r.pred_prob >= prob_threshold
    ]


@dataclass
class ExplanationSelection:
    n_total: int
    n_filtered: int
    explanation_class: int | None
    class_counts: dict[int, int]
    records: list[IndividualRecord]
    original_class: int | None


def select_explanation_phenotypes(
    records: list[IndividualRecord],
    *,
    prob_threshold: float = PROB_THRESHOLD,
    target_class: int | None = None,
    original_class: int | None = None,
) -> ExplanationSelection:
    """Pick high-confidence flip phenotypes toward an explanation class."""
    if not records:
        return ExplanationSelection(0, 0, None, {}, [], None)

    oc = original_class
    if oc is None or oc < 0:
        oc = records[0].original_class if records[0].original_class >= 0 else None

    flipped = filter_flipped_high_conf(
        records,
        prob_threshold=prob_threshold,
        original_class=oc,
    )
    dist = Counter(r.pred_class for r in flipped)

    if not dist:
        return ExplanationSelection(
            n_total=len(records),
            n_filtered=0,
            explanation_class=None,
            class_counts={},
            records=[],
            original_class=oc,
        )

    if target_class is None:
        explanation_class = dist.most_common(1)[0][0]
    else:
        explanation_class = int(target_class)

    selected = [r for r in flipped if r.pred_class == explanation_class]
    return ExplanationSelection(
        n_total=len(records),
        n_filtered=len(flipped),
        explanation_class=explanation_class,
        class_counts=dict(dist),
        records=selected,
        original_class=oc,
    )
