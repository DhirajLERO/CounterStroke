"""Resolve Graco per-sample directories."""

from __future__ import annotations

from pathlib import Path


def _sample_dir_names(sample_idx: int, y_true: int) -> list[str]:
    return [
        f"{sample_idx}_label_{y_true}",
        f"{sample_idx:05d}_label_{y_true}",
        f"{sample_idx:08d}_label_{y_true}",
    ]


def find_graco_sample_dir(graco_results_root: Path, sample_idx: int, y_true: int) -> Path | None:
    root = graco_results_root
    if not root.is_dir():
        return None

    for name in _sample_dir_names(sample_idx, y_true):
        exact = root / name
        if exact.is_dir():
            return exact

    patterns = [
        f"{sample_idx}_label_*",
        f"{sample_idx:05d}_label_*",
        f"{sample_idx:08d}_label_*",
    ]
    for pattern in patterns:
        for m in sorted(root.glob(pattern)):
            suffix = m.name.split("_label_", 1)[-1]
            try:
                y = int(suffix)
            except ValueError:
                continue
            if y == y_true:
                return m
    return None


def graco_sample_dir_hint(graco_results_root: Path, sample_idx: int, y_true: int) -> str:
    root = graco_results_root.resolve()
    lines = [f"graco results root: {root}"]

    if not root.is_dir():
        lines.append("  status: directory does not exist")
        lines.append("  fix: pass --graco-results PATH to your evolution output folder")
        return "\n".join(lines)

    expected = _sample_dir_names(sample_idx, y_true)[0]
    lines.append(f"  expected folder: {expected}/")

    for pattern in (f"{sample_idx}_label_*", f"{sample_idx:05d}_label_*"):
        matches = sorted(root.glob(pattern))
        if matches:
            lines.append(f"  folders for sample_idx {sample_idx}: {[p.name for p in matches[:8]]}")
            break
    else:
        examples = sorted(p.name for p in root.glob("*_label_*"))[:8]
        if examples:
            lines.append(f"  example folders in root: {examples}")
        else:
            lines.append("  no *_label_* folders found under this root")

    lines.append("  fix: confirm --sample-idx / --y-true match eval_manifest.json")
    lines.append("  fix: pass --graco-results PATH if results live elsewhere")
    return "\n".join(lines)


def run_dir(sample_dir: Path, run_id: int = 0) -> Path:
    return sample_dir / f"run_{run_id}"


def final_generation_csv(run_path: Path, run_id: int = 0) -> Path:
    return run_path / f"run_{run_id}_final_generation.csv"


def final_population_pickle(run_path: Path, run_id: int = 0) -> Path:
    return run_path / f"run_{run_id}_final_pop.pkl"


def summary_npz(run_path: Path, run_id: int = 0) -> Path:
    return run_path / f"run_{run_id}_summary.npz"
