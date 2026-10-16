"""Canonical sys.path setup for the Streamlit app."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_GRACO = _PKG.parent
_ANALYSIS = _GRACO / "analysis"
_EXPLAINABILITY = _GRACO / "explainability"


def setup_import_paths() -> tuple[Path, Path, Path]:
    """Ensure afp imports resolve to analyse_final_population/afp, not GracoImage/common."""
    if str(_PKG) not in sys.path:
        sys.path.insert(0, str(_PKG))
    for extra in (_ANALYSIS, _GRACO, _EXPLAINABILITY):
        s = str(extra)
        if s not in sys.path:
            sys.path.append(s)
    return _PKG, _GRACO, _ANALYSIS


def ensure_explainability_on_path() -> Path:
    """Make ``import explainability`` resolve to GracoImage/explainability."""
    s = str(_EXPLAINABILITY)
    if s not in sys.path:
        sys.path.append(s)
    return _EXPLAINABILITY
