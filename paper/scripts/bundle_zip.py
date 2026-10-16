#!/usr/bin/env python3
"""Stage 4: zip the bundle folder deterministically."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_common import (  # noqa: E402
    BUNDLE,
    MANIFEST_PATH,
    ROOT,
    read_manifest,
    rel,
    save_manifest,
    setup_logging,
    timestamp_slug,
    utc_now,
)

FIXED_DATE = (1980, 1, 1, 0, 0, 0)


def run(stamp: str | None = None, verbose: bool = False) -> dict:
    log = setup_logging(verbose)
    if not BUNDLE.exists():
        raise SystemExit(f"nothing to zip: {rel(BUNDLE)} does not exist")

    stamp = stamp or timestamp_slug()
    stamped = ROOT / f"results_bundle_{stamp}.zip"
    stable = ROOT / "results_bundle.zip"

    manifest = read_manifest()
    if manifest.get("stages", {}).pop("zip", None) is not None:
        save_manifest(manifest)

    files = sorted(p for p in BUNDLE.rglob("*") if p.is_file() and p.name != ".DS_Store")
    # manifest first so it is readable without extracting the whole archive
    ordered = [MANIFEST_PATH] if MANIFEST_PATH in files else []
    ordered += [p for p in files if p != MANIFEST_PATH]

    total = 0
    with zipfile.ZipFile(stamped, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in ordered:
            arcname = Path(BUNDLE.name) / path.relative_to(BUNDLE)
            info = zipfile.ZipInfo(str(arcname), date_time=FIXED_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            data = path.read_bytes()
            zf.writestr(info, data)
            total += len(data)

    stable.write_bytes(stamped.read_bytes())

    manifest["stages"]["zip"] = {
        "completed_utc": utc_now(),
        "archive": rel(stamped),
        "stable_archive": rel(stable),
        "n_files": len(ordered),
        "uncompressed_bytes": total,
        "compressed_bytes": stamped.stat().st_size,
    }
    save_manifest(manifest)

    log.info(
        "zipped %d files: %s (%.1f MiB -> %.1f MiB)",
        len(ordered),
        rel(stamped),
        total / 1024**2,
        stamped.stat().st_size / 1024**2,
    )
    log.info("stable copy: %s", rel(stable))
    return manifest["stages"]["zip"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", help="timestamp slug to use in the archive name")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    run(stamp=args.stamp, verbose=args.verbose)


if __name__ == "__main__":
    main()
