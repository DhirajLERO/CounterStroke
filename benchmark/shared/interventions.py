"""Sample do(parent) interventions (mirrors evaluate.produce_counterfactuals)."""

from __future__ import annotations

import numpy as np
import torch

rng = np.random.default_rng()


def different_value(possible_values, value, bins, attribute):
    if bins is not None and attribute in bins:
        return np.digitize(possible_values, bins[attribute]) != np.searchsorted(bins[attribute], value)
    return possible_values != value


def sample_intervention(
    factual_batch: dict,
    do_parent: str,
    dataset,
    device: str = "cuda",
) -> dict:
    possible_values = dataset.possible_values[do_parent]
    bins = getattr(dataset, "bins", None)
    values = factual_batch[do_parent].cpu()

    if do_parent not in ["digit", "apoE", "slice"]:
        interventions = {
            do_parent: torch.cat(
                [
                    torch.tensor(
                        np.random.choice(
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
                                torch.where((different_value(possible_values, value, bins, do_parent)).any(dim=1))
                            ],
                            axis=0,
                        )
                    ).unsqueeze(0)
                    for value in values
                ]
            ).to(device)
        }
    return interventions
