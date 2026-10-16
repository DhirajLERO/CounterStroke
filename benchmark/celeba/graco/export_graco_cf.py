"""Re-export Graco CF bundles (optional)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("outputs/graco/simple"))
    ap.add_argument("--dst", type=Path, default=Path("outputs/graco_export/simple"))
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(args.src.glob("cf_*.pt")):
        shutil.copy2(p, args.dst / p.name)
    print(f"Copied {len(list(args.src.glob('cf_*.pt')))} files to {args.dst}")


if __name__ == "__main__":
    main()
