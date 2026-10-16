"""PYTHONPATH and dependency checks for benchmark subprocesses."""

from __future__ import annotations

import os
import sys

from paths import BENCHMARK_ROOT, GRACOIMAGE_ROOT, VENDOR_PKG, setup_vendor_path


def benchmark_subprocess_env(extra: dict | None = None) -> dict:
    """Environment for child processes (vendor package on PYTHONPATH)."""
    setup_vendor_path()
    env = os.environ.copy()
    shared = str(BENCHMARK_ROOT / "shared")
    pp = str(VENDOR_PKG)
    env["PYTHONPATH"] = shared + os.pathsep + pp + os.pathsep + env.get("PYTHONPATH", "")
    if extra:
        env.update(extra)
    return env


def require_benchmark_deps() -> None:
    """Fail fast with install instructions if PyTorch is missing."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyTorch is not installed for the Python running this pipeline:\n"
            f"  {sys.executable}\n\n"
            "Activate the GracoImage environment and install dependencies:\n"
            f"  cd {GRACOIMAGE_ROOT}\n"
            "  source .venv/bin/activate    # or: conda activate <your-env>\n"
            "  pip install -r requirements.txt\n\n"
            "Re-run from the same environment (check with: which python3).\n"
            f"  cd {BENCHMARK_ROOT / 'morphomnist'}\n"
            "  python3 run_pipeline.py\n"
        )
