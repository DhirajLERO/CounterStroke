"""Fixed do(parent) intervention manifest for aligned SCM + Graco evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from interventions import different_value, sample_intervention


def _slot_seed(base_seed: int, sample_idx: int, do_parent: str) -> int:
    h = hash((int(base_seed), int(sample_idx), do_parent)) & 0xFFFFFFFF
    return int(h)


def sample_intervention_deterministic(
    factual_batch: dict,
    do_parent: str,
    dataset,
    *,
    seed: int,
    device: str = "cuda",
) -> dict:
    """Same rules as sample_intervention but with a fixed RNG per (idx, do_parent)."""
    rng = np.random.default_rng(seed)
    possible_values = dataset.possible_values[do_parent]
    bins = getattr(dataset, "bins", None)
    values = factual_batch[do_parent].cpu()

    if do_parent not in ["digit", "apoE", "slice"]:
        interventions = {
            do_parent: torch.cat(
                [
                    torch.tensor(
                        rng.choice(
                            possible_values[different_value(possible_values, value, bins, do_parent)]
                        )
                    ).unsqueeze(0)
                    for value in values
                ]
            )
            .view(-1)
            .unsqueeze(1)
            .to(device)
        }
    else:
        interventions = {
            do_parent: torch.cat(
                [
                    torch.tensor(
                        rng.choice(
                            possible_values[
                                torch.where(
                                    (different_value(possible_values, value, bins, do_parent)).any(dim=1)
                                )
                            ],
                            axis=0,
                        )
                    ).unsqueeze(0)
                    for value in values
                ]
            ).to(device)
        }
    return interventions


def _tensor_to_json(v: torch.Tensor):
    if v.numel() == 1:
        return float(v.item())
    return v.cpu().tolist()


def build_intervention_manifest(
    dataset,
    indices: list[int],
    attribute_size: dict,
    *,
    seed: int = 42,
    out_path: Path,
    transform=None,
) -> list[dict]:
    """Precompute interventions for every (sample_idx, do_parent) pair."""
    entries: list[dict] = []
    for idx in indices:
        batch = dataset[idx]
        if transform is not None and not isinstance(batch, dict):
            img, attrs = batch
            batch = transform(img, attrs)
        factual = {
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.ndim <= 1 else v)
            for k, v in batch.items()
            if k in attribute_size or k == "image"
        }
        if "image" in factual and factual["image"].ndim == 3:
            factual["image"] = factual["image"].unsqueeze(0)

        for do_parent in attribute_size:
            slot_seed = _slot_seed(seed, idx, do_parent)
            inter = sample_intervention_deterministic(
                factual, do_parent, dataset, seed=slot_seed, device="cpu"
            )
            entry = {
                "sample_idx": int(idx),
                "do_parent": do_parent,
                "seed": slot_seed,
                "intervention": {k: _tensor_to_json(v) for k, v in inter.items()},
            }
            entries.append(entry)

    payload = {"seed": seed, "entries": entries}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return entries


def load_intervention_manifest(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def manifest_intervention_tensors(entry: dict, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Restore intervention dict from manifest entry."""
    do_parent = entry["do_parent"]
    out = {}
    for k, v in entry["intervention"].items():
        if isinstance(v, list):
            t = torch.tensor(v, dtype=torch.float32)
            if do_parent in ("digit", "apoE", "slice") and len(v) > 1:
                t = t.view(1, -1)
            elif t.ndim == 1:
                t = t.view(1, -1)
            else:
                t = t.unsqueeze(0) if t.ndim == 1 else t
        else:
            t = torch.tensor([[float(v)]], dtype=torch.float32)
        out[k] = t.to(device)
    return out


def iter_manifest_entries(manifest: dict) -> list[dict]:
    return list(manifest.get("entries", []))


def lookup_manifest_entry(manifest: dict, sample_idx: int, do_parent: str) -> dict | None:
    for e in manifest.get("entries", []):
        if e["sample_idx"] == sample_idx and e["do_parent"] == do_parent:
            return e
    return None
