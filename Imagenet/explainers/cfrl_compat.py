"""Alibi CounterfactualRL checkpoint loading patch (PyTorch 2.6+ weights_only)."""
from __future__ import annotations

import torch


def patch_alibi_cfrl_pytorch_load_model() -> None:
    import alibi.explainers.backends.pytorch.cfrl_base as pt_cfrl

    if getattr(pt_cfrl, "_icf_cfrl_load_patched", False):
        return

    def load_model(path):  # type: ignore[no-untyped-def]
        model = torch.load(path, weights_only=False)
        model.eval()
        return model

    pt_cfrl.load_model = load_model  # type: ignore[assignment]
    pt_cfrl._icf_cfrl_load_patched = True  # type: ignore[attr-defined]
