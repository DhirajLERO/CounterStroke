"""Fixed test indices for SCM + Graco evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def build_eval_indices(
    dataset: Dataset,
    *,
    n_indices: int = 200,
    per_stratum: int | None = None,
    seed: int = 42,
    out_path: Path,
    stratify_fn=None,
) -> list[int]:
    """Build stratified or random index list for aligned SCM + Graco evaluation."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    target = min(n_indices, n)

    if stratify_fn is None:
        idxs = rng.choice(n, size=target, replace=False).tolist()
    else:
        buckets: dict[int, list[int]] = {}
        for i in range(n):
            key = int(stratify_fn(i))
            buckets.setdefault(key, []).append(i)
        n_strata = len(buckets)
        per = per_stratum if per_stratum is not None else max(1, target // max(n_strata, 1))
        idxs = []
        for key in sorted(buckets):
            pool = buckets[key]
            pick = rng.choice(pool, size=min(per, len(pool)), replace=False)
            idxs.extend(int(x) for x in pick)
        if len(idxs) > target:
            idxs = rng.choice(idxs, size=target, replace=False).tolist()
        else:
            idxs = sorted(idxs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"indices": idxs, "seed": seed, "n_indices": len(idxs)}, indent=2))
    return idxs


def load_eval_indices(path: Path | str) -> list[int]:
    data = json.loads(Path(path).read_text())
    return list(data["indices"])


def subset_dataset(dataset: Dataset, indices_path: Path | str) -> Subset:
    return Subset(dataset, load_eval_indices(indices_path))


def morphomnist_stratify(dataset) -> callable:
    def fn(i: int) -> int:
        item = dataset[i]
        if isinstance(item, dict) and "digit" in item:
            d = item["digit"]
            if d.ndim > 0:
                return int(d.argmax())
            return int(d.item())
        return i % 10

    return fn


def _attr_bucket(item: dict, attr: str) -> int:
    if attr not in item:
        return 0
    v = item[attr]
    if isinstance(v, torch.Tensor):
        if v.numel() > 1:
            return int(v.argmax().item())
        return int(round(float(v.item())))
    return int(v)


def celeba_simple_stratify(dataset) -> callable:
    def fn(i: int) -> int:
        item = dataset[i]
        if not isinstance(item, dict):
            return i % 2
        return _attr_bucket(item, "Smiling")

    return fn


def celeba_complex_stratify(dataset) -> callable:
    def fn(i: int) -> int:
        item = dataset[i]
        if not isinstance(item, dict):
            return i % 2
        return _attr_bucket(item, "Young")

    return fn
