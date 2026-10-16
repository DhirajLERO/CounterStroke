#!/usr/bin/env python3
"""Build the results bundle: gather -> consolidate -> stats -> zip."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bundle_consolidate  # noqa: E402
import bundle_gather  # noqa: E402
import bundle_stats  # noqa: E402
import bundle_zip  # noqa: E402
from bundle_common import (  # noqa: E402
    BUNDLE,
    RAW,
    rel,
    setup_logging,
    timestamp_slug,
)

STAGES = ("gather", "consolidate", "stats", "zip")


def run(
    stage: str = "all",
    include_tensors: bool = False,
    clean: bool = False,
    skip_zip: bool = False,
    verbose: bool = False,
) -> None:
    log = setup_logging(verbose)
    requested = STAGES if stage == "all" else (stage,)
    stamp = timestamp_slug()
    started = time.time()

    if "gather" not in requested and not RAW.exists():
        raise SystemExit(
            f"{rel(RAW)} is missing; run --stage gather first (or --stage all)"
        )

    for name in requested:
        if name == "zip":
            continue
        log.info("=== stage: %s ===", name)
        t0 = time.time()
        if name == "gather":
            bundle_gather.run(include_tensors=include_tensors, clean=clean, verbose=verbose)
        elif name == "consolidate":
            bundle_consolidate.run(verbose=verbose)
        elif name == "stats":
            bundle_stats.run(verbose=verbose)
        log.info("stage %s finished in %.1fs", name, time.time() - t0)

    if not skip_zip:
        log.info("=== stage: zip ===")
        bundle_zip.run(stamp=stamp, verbose=verbose)

    log.info("bundle at %s (total %.1fs)", rel(BUNDLE), time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=("all", *STAGES),
        help="stage to run (default: all)",
    )
    parser.add_argument(
        "--include-tensors",
        action="store_true",
        help="gather per-sample tensors.npz as well (large)",
    )
    parser.add_argument("--clean", action="store_true", help="delete raw/ before gathering")
    parser.add_argument("--skip-zip", action="store_true", help="do not rebuild the archive")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    run(
        stage=args.stage,
        include_tensors=args.include_tensors,
        clean=args.clean,
        skip_zip=args.skip_zip,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
