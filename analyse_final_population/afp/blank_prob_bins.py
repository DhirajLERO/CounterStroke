"""Pick blank-canvas population members by confidence bins (sparsity-first)."""

from __future__ import annotations

from dataclasses import dataclass

from .population_io import IndividualRecord

# Half-open bins except the last, which includes 1.0.
PROB_BINS: tuple[tuple[float, float], ...] = (
    (0.5, 0.6),
    (0.6, 0.7),
    (0.7, 0.8),
    (0.8, 0.9),
    (0.9, 1.0),
)


@dataclass(frozen=True)
class BinPick:
    lo: float
    hi: float
    record: IndividualRecord | None

    @property
    def label(self) -> str:
        return f"{int(self.lo * 100)}–{int(self.hi * 100)}%"


def _in_bin(prob: float, lo: float, hi: float, *, last: bool) -> bool:
    if last:
        return lo <= prob <= hi + 1e-9
    return lo <= prob < hi


def _sort_key(r: IndividualRecord) -> tuple[float, float]:
    o2 = float(r.o2) if r.o2 is not None else float("inf")
    o3 = float(r.o3) if r.o3 is not None else float("inf")
    return (o2, o3)


def pick_records_by_prob_bins(
    records: list[IndividualRecord],
    target_class: int,
    bins: tuple[tuple[float, float], ...] = PROB_BINS,
) -> list[BinPick]:
    """One sparsity-first individual per bin with ``pred_class == target_class``."""
    out: list[BinPick] = []
    n = len(bins)
    for i, (lo, hi) in enumerate(bins):
        last = i == n - 1
        candidates = [
            r
            for r in records
            if r.pred_class == target_class and _in_bin(r.pred_prob, lo, hi, last=last)
        ]
        best = min(candidates, key=_sort_key) if candidates else None
        out.append(BinPick(lo=lo, hi=hi, record=best))
    return out
