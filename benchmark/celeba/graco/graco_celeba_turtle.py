"""CLI entry for CelebA Graco evolution (see ../run_graco_evolve.py)."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_graco_evolve.py"), run_name="__main__")
