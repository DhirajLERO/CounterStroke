"""Shared eval protocol helpers for aligned SCM + Graco benchmark runs."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset, Subset

from eval_interventions import iter_manifest_entries, load_intervention_manifest, manifest_intervention_tensors
from eval_subset import load_eval_indices, subset_dataset

GRACO_VARIANTS = ("graco_base_o2", "graco_base_o3", "graco50")

DEFAULT_EVAL_SAMPLES = 200


def expected_cf_count(n_indices: int, n_parents: int) -> int:
    return n_indices * n_parents


def load_eval_subset(dataset: Dataset, indices_path: Path) -> Subset:
    return subset_dataset(dataset, indices_path)


def iter_eval_slots(
    indices: list[int],
    attribute_size: dict,
    manifest_path: Path | None = None,
    manifest: dict | None = None,
):
    """Yield (sample_idx, do_parent, intervention_entry) for each eval slot."""
    if manifest is None and manifest_path is not None:
        manifest = load_intervention_manifest(manifest_path)
    index_set = set(indices)
    parents = list(attribute_size.keys())

    if manifest is not None:
        for entry in iter_manifest_entries(manifest):
            if entry["sample_idx"] in index_set:
                yield entry["sample_idx"], entry["do_parent"], entry
        return

    for idx in indices:
        for do_parent in parents:
            yield idx, do_parent, None


def interventions_path_for_indices(indices_path: Path) -> Path:
    return indices_path.parent / indices_path.name.replace("eval_indices", "eval_interventions")
