"""Load benchmark anti-causal predictors (same as evaluate.py)."""

from __future__ import annotations

import os
from json import load as json_load
from pathlib import Path
from typing import Any

import torch

from paths import setup_vendor_path

setup_vendor_path()

from models.classifiers.classifier import Classifier  # noqa: E402
from models.classifiers.celeba_classifier import CelebaClassifier  # noqa: E402
from models.classifiers.celeba_complex_classifier import CelebaComplexClassifier  # noqa: E402
from datasets.transforms import get_attribute_ids  # noqa: E402


def load_predictors(classifier_config_path: Path, scm_config: dict[str, Any], device: str = "cuda"):
    with open(classifier_config_path) as f:
        config_cls = json_load(f)

    dataset = scm_config["dataset"]
    attribute_size = scm_config["attribute_size"]

    if dataset == "morphomnist":
        predictors = {
            atr: Classifier(
                attr=atr,
                num_outputs=attribute_size[atr],
                context_dim=len(list(config_cls["anticausal_graph"][atr])),
            )
            for atr in attribute_size.keys()
        }
    elif dataset == "celeba":
        if sum(attribute_size.values()) == 4:
            predictors = {
                atr: CelebaComplexClassifier(
                    attr=atr,
                    context_dim=len(list(config_cls["anticausal_graph"][atr])),
                    num_outputs=config_cls.get(f"{atr}_num_out", 1),
                    lr=config_cls.get("lr", 1e-4),
                    version=config_cls.get("version", "standard"),
                )
                for atr in attribute_size.keys()
            }
        else:
            predictors = {
                atr: CelebaClassifier(
                    attr=atr,
                    num_outputs=config_cls.get(f"{atr}_num_out", 1),
                    lr=config_cls.get("lr", 1e-3),
                )
                for atr in attribute_size.keys()
            }
    else:
        raise ValueError(dataset)

    ckpt_path = config_cls["ckpt_path"]
    if not ckpt_path.endswith("/"):
        ckpt_path += "/"

    for key, clf in predictors.items():
        file_name = next((f for f in os.listdir(ckpt_path) if f.startswith(key)), None)
        if file_name is None:
            raise FileNotFoundError(f"No checkpoint for {key} in {ckpt_path}")
        clf.load_state_dict(torch.load(ckpt_path + file_name, map_location=device)["state_dict"])
        clf.to(device)
        clf.eval()

    return predictors, config_cls
