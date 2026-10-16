"""PageRank backend parity tests (implicit vs scipy vs networkx)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from explainability.graph.pagerank import pagerank_cooccur_map  # noqa: E402


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    fa = a.ravel().astype(np.float64)
    fb = b.ravel().astype(np.float64)
    if fa.std() < 1e-12 or fb.std() < 1e-12:
        return 1.0 if np.allclose(fa, fb) else 0.0
    return float(np.corrcoef(fa, fb)[0, 1])


def _synthetic_diffs(shape: tuple[int, int]) -> list[np.ndarray]:
    diffs = [
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
    ]
    diffs[0][2, 2] = 0.5
    diffs[0][2, 3] = 0.4
    diffs[1][2, 2] = 0.6
    diffs[1][3, 3] = 0.3
    diffs[2][4, 4] = 0.8
    diffs[2][4, 5] = 0.2
    return diffs


def test_empty_diffs():
    shape = (8, 8)
    out = pagerank_cooccur_map([], shape, backend="implicit")
    assert out.shape == shape
    assert np.all(out == 0)


def test_scipy_matches_networkx_on_synthetic():
    pytest.importorskip("networkx")
    shape = (8, 8)
    diffs = _synthetic_diffs(shape)

    scipy_out = pagerank_cooccur_map(diffs, shape, backend="scipy")
    nx_out = pagerank_cooccur_map(diffs, shape, backend="networkx")

    assert scipy_out.shape == shape
    assert nx_out.shape == shape
    assert _pearson(scipy_out, nx_out) > 0.99
    assert scipy_out.argmax() == nx_out.argmax()


def test_implicit_matches_scipy_on_synthetic():
    shape = (8, 8)
    diffs = _synthetic_diffs(shape)

    implicit_out = pagerank_cooccur_map(diffs, shape, backend="implicit")
    scipy_out = pagerank_cooccur_map(diffs, shape, backend="scipy")

    assert implicit_out.shape == shape
    assert _pearson(implicit_out, scipy_out) > 0.99
    assert implicit_out.argmax() == scipy_out.argmax()


def test_implicit_matches_scipy_on_dense_patch():
    shape = (16, 16)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(8):
        d = np.zeros(shape, dtype=np.float32)
        patch = rng.uniform(0.1, 1.0, size=(6, 6)).astype(np.float32)
        d[3:9, 3:9] = patch
        diffs.append(d)

    implicit_out = pagerank_cooccur_map(diffs, shape, backend="implicit")
    scipy_out = pagerank_cooccur_map(diffs, shape, backend="scipy")

    assert _pearson(implicit_out, scipy_out) > 0.99
    assert implicit_out.argmax() == scipy_out.argmax()


def test_single_pixel_no_edges():
    shape = (6, 6)
    d = np.zeros(shape, dtype=np.float32)
    d[1, 1] = 0.7
    implicit_out = pagerank_cooccur_map([d], shape, backend="implicit")
    scipy_out = pagerank_cooccur_map([d], shape, backend="scipy")
    assert implicit_out[1, 1] > 0.9
    assert scipy_out[1, 1] > 0.9
