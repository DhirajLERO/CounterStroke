#!/usr/bin/env python3
"""Stage 1: copy every existing result artifact into results_bundle/raw."""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bundle_common import (  # noqa: E402
    BUNDLE,
    KNOWN_GAPS,
    RAW,
    ROOT,
    manifest_entry,
    read_manifest,
    rel,
    save_manifest,
    setup_logging,
    utc_now,
    write_text,
)


@dataclass
class Source:
    """One declarative gather rule."""

    dest: str
    patterns: list[str]
    description: str
    strip: str = ""
    optional: bool = False
    tag: str = ""
    notes: list[str] = field(default_factory=list)


SOURCES: list[Source] = [
    Source(
        dest="raw/main",
        patterns=["paper_results_main/*.csv", "paper_results_main/MANIFEST.json"],
        strip="paper_results_main",
        description=(
            "Pre-aggregated per-sample metrics, metric summaries, validity summary, "
            "paired and batch distributional metrics. Present on dev machines only. "
            "On the cluster the consolidation stage reads from raw/generation instead."
        ),
        optional=True,
    ),
    Source(
        dest="raw/explainability",
        patterns=["paper_results_main/explainability/*.csv"],
        strip="paper_results_main/explainability",
        description="Pre-aggregated per-sample SRG scores (dev machine only).",
        optional=True,
    ),
    Source(
        dest="raw/benchmark",
        patterns=[
            # dev machine: pre-aggregated under paper_results_main/
            "paper_results_main/benchmark/*.csv",
            # cluster: written by benchmark/shared/aggregate_results.py
            "benchmark/morphomnist/outputs/summary.csv",
            "benchmark/celeba/outputs/summary.csv",
            # alternative names used by some pipeline versions
            "benchmark/morphomnist/morphomnist_summary.csv",
            "benchmark/celeba/celeba_summary.csv",
            "benchmark/morphomnist/outputs/morphomnist_summary.csv",
            "benchmark/celeba/outputs/celeba_summary.csv",
        ],
        strip="paper_results_main/benchmark",
        description=(
            "Deep-SCM (MorphoMNIST, CelebA) aggregated benchmark summaries. "
            "On the cluster these are produced by benchmark/shared/aggregate_results.py."
        ),
        optional=True,
    ),
    Source(
        dest="raw/figures",
        patterns=["paper_results_main/figures/*.pdf", "paper_results_main/figures/*.png"],
        strip="paper_results_main/figures",
        description="Distribution and SRG curve figures (dev machine only).",
        optional=True,
    ),
    Source(
        dest="raw/generation",
        patterns=[
            "MNIST/outputs/generation/per_sample_metrics.csv",
            "Cifar10/outputs/generation/per_sample_metrics.csv",
            "Imagenet/outputs/generation/per_sample_metrics.csv",
            "MNIST/outputs/generation/distribution_metrics.csv",
            "Cifar10/outputs/generation/distribution_metrics.csv",
            "Imagenet/outputs/generation/distribution_metrics.csv",
            "MNIST/outputs/generation/batch_fid.csv",
            "Cifar10/outputs/generation/batch_fid.csv",
            "Imagenet/outputs/generation/batch_fid.csv",
            "MNIST/outputs/generation/cmmd_per_sample.csv",
            "Cifar10/outputs/generation/cmmd_per_sample.csv",
            "Imagenet/outputs/generation/cmmd_per_sample.csv",
            "MNIST/outputs/generation/cmmd_batch_stratified.csv",
            "Cifar10/outputs/generation/cmmd_batch_stratified.csv",
            "Imagenet/outputs/generation/cmmd_batch_stratified.csv",
        ],
        strip="outputs/generation",
        description=(
            "Per-dataset result CSVs written directly by the generation pipeline. "
            "This is the primary source on the cluster where paper_results_main/ "
            "does not exist. Absent on dev machines that only have the aggregated "
            "files; both sources are optional so the script works in either env."
        ),
        optional=True,
    ),
    Source(
        dest="raw/srg_curves",
        patterns=["explainability/outputs/*/sample_*/srg/*.npz"],
        strip="explainability/outputs",
        description=(
            "Per-sample MIF/LIF corruption curves. Required to recompute "
            "AUC(MIF)/AUC(LIF), which are currently hardcoded in the paper."
        ),
        tag="srg_curves",
    ),
    Source(
        dest="raw/config",
        patterns=[
            "MNIST/config.json",
            "Cifar10/config.json",
            "Imagenet/config.json",
            "benchmark/morphomnist/graco_config.json",
            "benchmark/celeba/graco_config_simple.json",
            "benchmark/celeba/graco_config_complex.json",
            "METRICS.md",
        ],
        description=(
            "Search configurations and metric definitions. These pin population "
            "size and generation count, which the paper's configuration table "
            "reports differently from the checked-in configs."
        ),
    ),
    Source(
        dest="raw/blankcanvas",
        patterns=["paper/figures/blankcanvas/**/*.png"],
        strip="paper/figures/blankcanvas",
        description="Blank-canvas evolved sketches, SSIM references and co-change maps.",
    ),
    Source(
        dest="raw/generation",
        patterns=[
            "MNIST/outputs/results/generation/summary_tables.csv",
            "Cifar10/outputs/results/generation/summary_tables.csv",
            "Imagenet/outputs/results/generation/summary_tables.csv",
        ],
        strip="outputs/results/generation",
        description=(
            "Long-format aggregated summary produced by common/plot_methods.py "
            "(method × metric × mean/std/n, successful flips only). Provides a "
            "fast cross-check of consolidated metric_summary.csv values."
        ),
        optional=True,
    ),
    Source(
        dest="raw/generation_meta",
        patterns=[
            "MNIST/outputs/generation/sample_*/meta.json",
            "Cifar10/outputs/generation/sample_*/meta.json",
            "Imagenet/outputs/generation/sample_*/meta.json",
        ],
        description=(
            "Per-sample generation metadata, including the selected Pareto row "
            "and its objective values for each CounterStroke readout."
        ),
        optional=True,
    ),
    Source(
        dest="raw/explainability_meta",
        patterns=["explainability/outputs/*/sample_*/meta.json"],
        strip="explainability/outputs",
        description=(
            "Per-sample attribution metadata (explanation class, phenotype "
            "counts, selected individual)."
        ),
        optional=True,
    ),
    Source(
        dest="raw/benchmark_outputs",
        patterns=[
            "benchmark/morphomnist/outputs/**/*.json",
            "benchmark/morphomnist/outputs/**/*.txt",
            "benchmark/morphomnist/outputs/**/*.csv",
            "benchmark/celeba/outputs/**/*.json",
            "benchmark/celeba/outputs/**/*.txt",
            "benchmark/celeba/outputs/**/*.csv",
        ],
        strip="benchmark",
        description=(
            "Raw Deep-SCM benchmark outputs (per-model metrics.txt, "
            "graco_metrics.json, merged summary.csv) behind the aggregated "
            "summaries."
        ),
        optional=True,
    ),
    Source(
        dest="raw/blankcanvas_outputs",
        patterns=[
            "analysis/*/outputs/**/*.csv",
            "analysis/*/outputs/**/*.json",
        ],
        strip="analysis",
        description=(
            "Blank-canvas synthesis outputs (per-target summaries and evolved "
            "individual records) needed for a quantitative blank-canvas table."
        ),
        optional=True,
    ),
    Source(
        dest="raw/tensors",
        patterns=[
            "MNIST/outputs/generation/sample_*/tensors.npz",
            "Cifar10/outputs/generation/sample_*/tensors.npz",
            "Imagenet/outputs/generation/sample_*/tensors.npz",
        ],
        description=(
            "Per-sample counterfactual tensors for qualitative grids and any "
            "recomputation of set-level metrics on flip-only subsets. Large: "
            "gathered only when --include-tensors is passed."
        ),
        optional=True,
        tag="tensors",
    ),
]



def _matches(pattern: str) -> list[Path]:
    return sorted(p for p in ROOT.glob(pattern) if p.is_file())


def _destination(spec: Source, source: Path) -> Path:
    """Destination path inside the bundle."""
    relative = source.relative_to(ROOT)
    if spec.strip:
        strip_path = Path(spec.strip)
        try:
            # standard case: strip is a leading prefix
            relative = relative.relative_to(strip_path)
        except ValueError:
            # fallback: remove strip_path from somewhere inside the tree by
            # rebuilding the path without that segment.  For example,
            #   MNIST/outputs/generation/file.csv, strip=outputs/generation
            # → MNIST/file.csv
            parts = relative.parts
            strip_parts = strip_path.parts
            n = len(strip_parts)
            for i in range(len(parts) - n):
                if parts[i : i + n] == strip_parts:
                    relative = Path(*parts[:i]) / Path(*parts[i + n :])
                    break
            else:
                relative = Path(relative.name)
    return BUNDLE / spec.dest / relative


def run(include_tensors: bool = False, clean: bool = False, verbose: bool = False) -> dict:
    log = setup_logging(verbose)
    if clean and RAW.exists():
        log.info("removing existing %s", rel(RAW))
        shutil.rmtree(RAW)
    RAW.mkdir(parents=True, exist_ok=True)

    copied: list[dict] = []
    missing: list[dict] = []
    per_source: dict[str, dict] = {}

    for spec in SOURCES:
        if spec.tag == "tensors" and not include_tensors:
            missing.append(
                {
                    "dest": spec.dest,
                    "patterns": spec.patterns,
                    "description": spec.description,
                    "optional": True,
                    "reason": "skipped: pass --include-tensors to gather these",
                }
            )
            continue

        found: list[Path] = []
        for pattern in spec.patterns:
            found.extend(_matches(pattern))
        found = sorted(set(found))

        if not found:
            if spec.optional:
                log.info("%-22s absent here (expected only where produced)", spec.dest)
            else:
                log.warning("no matches for %s (%s)", spec.dest, ", ".join(spec.patterns))
            missing.append(
                {
                    "dest": spec.dest,
                    "patterns": spec.patterns,
                    "description": spec.description,
                    "optional": spec.optional,
                    "reason": "no files matched in this checkout",
                }
            )
            continue

        n_bytes = 0
        for source in found:
            destination = _destination(spec, source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            entry = manifest_entry(destination, source=source, stage="gather")
            entry["group"] = spec.dest
            copied.append(entry)
            n_bytes += entry["bytes"]

        per_source[spec.dest] = {
            "description": spec.description,
            "patterns": spec.patterns,
            "n_files": len(found),
            "bytes": n_bytes,
        }
        log.info("%-22s %4d files, %8.1f KiB", spec.dest, len(found), n_bytes / 1024)

    manifest = read_manifest()
    manifest["files"] = [f for f in manifest.get("files", []) if f.get("stage") != "gather"]
    manifest["files"].extend(copied)
    manifest["missing_sources"] = missing
    manifest["known_gaps"] = KNOWN_GAPS
    manifest["stages"]["gather"] = {
        "completed_utc": utc_now(),
        "n_files": len(copied),
        "bytes": int(sum(e["bytes"] for e in copied)),
        "include_tensors": include_tensors,
        "groups": per_source,
    }
    save_manifest(manifest)
    _write_readme(per_source, missing)

    log.info(
        "gather complete: %d files, %.1f MiB",
        len(copied),
        sum(e["bytes"] for e in copied) / 1024**2,
    )
    return manifest["stages"]["gather"]


def _write_readme(per_source: dict[str, dict], missing: list[dict]) -> Path:
    lines = [
        "# CounterStroke results bundle",
        "",
        "Self-contained snapshot of every result artifact behind the Results "
        "section, plus consolidated tables and statistical tests derived from them.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python paper/scripts/make_bundle.py --stage all",
        "```",
        "",
        "## Layout",
        "",
        "- `raw/` verbatim copies of the source artifacts, with SHA-256 provenance "
        "in `BUNDLE_MANIFEST.json`",
        "- `consolidated/` tidy tables with canonical method names and explicit "
        "evaluation subsets",
        "- `stats/` statistical tests, rankings, diagnostics and `STATS_REPORT.md`",
        "",
        "## Gathered sources",
        "",
    ]
    for dest, info in per_source.items():
        lines.append(f"### `{dest}`")
        lines.append("")
        lines.append(info["description"])
        lines.append("")
        lines.append(f"- files: {info['n_files']}")
        lines.append(f"- size: {info['bytes'] / 1024:.1f} KiB")
        lines.append(f"- patterns: {', '.join('`' + p + '`' for p in info['patterns'])}")
        lines.append("")

    if missing:
        lines += ["## Sources not present in this run", ""]
        for item in missing:
            kind = "optional" if item.get("optional") else "expected"
            lines.append(f"- `{item['dest']}` ({kind}): {item['reason']}")
        lines.append("")
        lines.append(
            "Optional sources exist only where the corresponding pipeline ran; "
            "re-running gather there picks them up with no other change."
        )
        lines.append("")

    lines += ["## Known gaps", ""]
    lines += [f"- {gap}" for gap in KNOWN_GAPS]
    lines.append("")

    return write_text("\n".join(lines), BUNDLE / "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-tensors",
        action="store_true",
        help="also copy per-sample tensors.npz (large; absent in this checkout)",
    )
    parser.add_argument("--clean", action="store_true", help="delete raw/ before copying")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    run(include_tensors=args.include_tensors, clean=args.clean, verbose=args.verbose)


if __name__ == "__main__":
    main()
