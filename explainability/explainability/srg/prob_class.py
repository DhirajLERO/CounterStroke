"""Which class probability SRG tracks (model base class vs Graco explanation class)."""

from __future__ import annotations

GRACO_SRG_PROB_MODES = ("base", "pred")
# CLI / legacy YAML alias; normalized to "base" in parse_graco_srg_prob_mode.
GRACO_SRG_PROB_MODE_ALIASES = {"true": "base", "base": "base", "pred": "pred"}
CLI_GRACO_SRG_PROB_MODES = ("base", "pred", "true")


def parse_graco_srg_prob_mode(name: str) -> str:
    mode = GRACO_SRG_PROB_MODE_ALIASES.get(str(name).lower())
    if mode is None:
        valid = ", ".join(GRACO_SRG_PROB_MODES)
        raise ValueError(f"srg_graco_prob_class must be one of {valid} (true aliases base), got {name!r}")
    return mode


def srg_prob_class_for_method(
    method: str,
    *,
    base_class: int,
    explanation_class: int,
    graco_methods: set[str],
    graco_mode: str,
) -> int:
    """All methods track model argmax on original; Graco maps may switch to explanation class."""
    if method in graco_methods and parse_graco_srg_prob_mode(graco_mode) == "pred":
        return explanation_class
    return base_class
