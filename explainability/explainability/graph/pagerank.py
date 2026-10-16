"""Co-occurrence graph PageRank over edited pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..phenotype.aggregations import normalize_01
from ..phenotype.diffs import EPS


@dataclass(frozen=True)
class CooccurGraph:
    """Sparse co-edit graph over edited pixel coordinates."""

    node_coords: list[tuple[int, int]]
    node_intensity: list[float]
    edges: list[tuple[int, int, float]]


@dataclass(frozen=True)
class IncidenceData:
    """CF incidence lists for implicit co-occurrence matvecs."""

    node_coords: list[tuple[int, int]]
    intensities: np.ndarray
    cf_node_indices: list[np.ndarray]


def _build_cooccur_graph(
    diffs: list[np.ndarray],
    shape: tuple[int, int],
    *,
    eps: float = EPS,
) -> CooccurGraph | None:
    node_intensity: dict[tuple[int, int], list[float]] = {}
    edge_counts: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}

    for d in diffs:
        mask = d > eps
        coords = list(zip(*np.nonzero(mask)))
        if not coords:
            continue
        for r, c in coords:
            node_intensity.setdefault((r, c), []).append(float(d[r, c]))
        if len(coords) < 2:
            continue
        for i, p in enumerate(coords):
            for q in coords[i + 1 :]:
                key = (p, q) if p < q else (q, p)
                edge_counts[key] = edge_counts.get(key, 0.0) + 1.0

    if not node_intensity:
        return None

    coord_to_idx = {coord: i for i, coord in enumerate(node_intensity.keys())}
    intensities = [float(np.mean(vals)) for vals in node_intensity.values()]
    node_coords = list(node_intensity.keys())

    edges: list[tuple[int, int, float]] = []
    for (p, q), w in edge_counts.items():
        ip = float(np.mean(node_intensity[p]))
        iq = float(np.mean(node_intensity[q]))
        weight = w * float(np.sqrt(max(ip, 0) * max(iq, 0)))
        i = coord_to_idx[p]
        j = coord_to_idx[q]
        edges.append((i, j, weight))

    return CooccurGraph(node_coords=node_coords, node_intensity=intensities, edges=edges)


def _build_incidence_data(
    diffs: list[np.ndarray],
    shape: tuple[int, int],
    *,
    eps: float = EPS,
) -> IncidenceData | None:
    """Build per-CF node index lists without materialising all-pairs edges."""
    intensity_sum: dict[tuple[int, int], float] = {}
    intensity_count: dict[tuple[int, int], int] = {}
    cf_coords: list[list[tuple[int, int]]] = []

    for d in diffs:
        mask = d > eps
        if not np.any(mask):
            continue
        coords_this: list[tuple[int, int]] = []
        for r, c in zip(*np.nonzero(mask)):
            coord = (int(r), int(c))
            val = float(d[r, c])
            intensity_sum[coord] = intensity_sum.get(coord, 0.0) + val
            intensity_count[coord] = intensity_count.get(coord, 0) + 1
            coords_this.append(coord)
        if coords_this:
            cf_coords.append(coords_this)

    if not intensity_sum:
        return None

    node_coords = list(intensity_sum.keys())
    coord_to_idx = {coord: i for i, coord in enumerate(node_coords)}
    intensities = np.array(
        [intensity_sum[c] / max(intensity_count[c], 1) for c in node_coords],
        dtype=np.float64,
    )
    cf_node_indices = [
        np.fromiter((coord_to_idx[c] for c in coords), dtype=np.int32, count=len(coords))
        for coords in cf_coords
    ]
    return IncidenceData(
        node_coords=node_coords,
        intensities=intensities,
        cf_node_indices=cf_node_indices,
    )


def _matvec_cooccur(u: np.ndarray, cf_node_indices: list[np.ndarray]) -> np.ndarray:
    """Apply off-diagonal co-occurrence adjacency: (C @ u) without self-pairs."""
    out = np.zeros_like(u, dtype=np.float64)
    for idx in cf_node_indices:
        if idx.size < 2:
            continue
        uidx = u[idx]
        s = float(uidx.sum())
        out[idx] += s - uidx
    return out


def _weighted_matvec(
    x: np.ndarray,
    f: np.ndarray,
    cf_node_indices: list[np.ndarray],
) -> np.ndarray:
    return f * _matvec_cooccur(f * x, cf_node_indices)


def _intensity_only_map_from_nodes(
    node_coords: list[tuple[int, int]],
    intensities: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    for (r, c), intensity in zip(node_coords, intensities):
        out[r, c] = float(intensity)
    return normalize_01(out)


def _intensity_only_map(graph: CooccurGraph, shape: tuple[int, int]) -> np.ndarray:
    return _intensity_only_map_from_nodes(graph.node_coords, np.asarray(graph.node_intensity), shape)


def _scatter_pagerank_coords(
    node_coords: list[tuple[int, int]],
    scores: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    for (r, c), score in zip(node_coords, scores):
        out[r, c] = float(score)
    return normalize_01(out)


def _scatter_pagerank(
    graph: CooccurGraph,
    scores: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    return _scatter_pagerank_coords(graph.node_coords, scores, shape)


def _pagerank_implicit(
    data: IncidenceData,
    shape: tuple[int, int],
    *,
    damping: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """PageRank via implicit B @ B^T matvecs (same graph/weights as explicit build)."""
    n = len(data.node_coords)
    if n == 0:
        return np.zeros(shape, dtype=np.float32)

    f = np.sqrt(np.maximum(data.intensities, 0.0))
    has_edges = any(idx.size >= 2 for idx in data.cf_node_indices)
    if not has_edges:
        return _intensity_only_map_from_nodes(data.node_coords, data.intensities, shape)

    out_strength = _weighted_matvec(np.ones(n, dtype=np.float64), f, data.cf_node_indices)
    dangling = out_strength == 0.0
    out_strength_safe = out_strength.copy()
    out_strength_safe[dangling] = 1.0

    p = np.repeat(1.0 / n, n)
    dangling_weights = p.copy()
    x = p.copy()
    for _ in range(max_iter):
        xlast = x.copy()
        wx = _weighted_matvec(x, f, data.cf_node_indices)
        step = wx / out_strength_safe
        dangling_sum = float(np.sum(x[dangling]))
        x = damping * (step + dangling_sum * dangling_weights) + (1.0 - damping) * p
        if np.abs(x - xlast).sum() < tol:
            break

    return _scatter_pagerank_coords(data.node_coords, x, shape)


def _pagerank_networkx(graph: CooccurGraph, shape: tuple[int, int], *, damping: float) -> np.ndarray:
    import networkx as nx

    if not graph.edges:
        return _intensity_only_map(graph, shape)

    g = nx.Graph()
    for coord, intensity in zip(graph.node_coords, graph.node_intensity):
        g.add_node(coord, intensity=intensity)
    for i, j, weight in graph.edges:
        g.add_edge(graph.node_coords[i], graph.node_coords[j], weight=weight)

    pr = nx.pagerank(g, alpha=damping, weight="weight")
    out = np.zeros(shape, dtype=np.float32)
    for (r, c), score in pr.items():
        out[r, c] = float(score)
    return normalize_01(out)


def _pagerank_scipy(
    graph: CooccurGraph,
    shape: tuple[int, int],
    *,
    damping: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Weighted PageRank via scipy sparse mats (matches NetworkX ``_pagerank_scipy``)."""
    from scipy import sparse

    if not graph.edges:
        return _intensity_only_map(graph, shape)

    n = len(graph.node_coords)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for i, j, weight in graph.edges:
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([weight, weight])

    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    out_strength = np.asarray(adj.sum(axis=1)).ravel()
    dangling = out_strength == 0.0
    out_strength[dangling] = 1.0
    q = (sparse.diags(1.0 / out_strength) @ adj).T

    p = np.repeat(1.0 / n, n)
    dangling_weights = p.copy()
    x = p.copy()
    for _ in range(max_iter):
        xlast = x.copy()
        dangling_sum = float(np.sum(x[dangling]))
        x = damping * (x @ q + dangling_sum * dangling_weights) + (1.0 - damping) * p
        if np.abs(x - xlast).sum() < tol:
            break

    return _scatter_pagerank(graph, x, shape)


def pagerank_cooccur_map(
    diffs: list[np.ndarray],
    shape: tuple[int, int],
    *,
    eps: float = EPS,
    damping: float = 0.85,
    backend: str = "implicit",
) -> np.ndarray:
    """Build H×W importance from pixel co-occurrence PageRank."""
    if not diffs:
        return np.zeros(shape, dtype=np.float32)

    backend_norm = str(backend).lower().strip()
    if backend_norm == "implicit":
        data = _build_incidence_data(diffs, shape, eps=eps)
        if data is None:
            return np.zeros(shape, dtype=np.float32)
        return _pagerank_implicit(data, shape, damping=damping)

    graph = _build_cooccur_graph(diffs, shape, eps=eps)
    if graph is None:
        return np.zeros(shape, dtype=np.float32)

    if backend_norm == "networkx":
        return _pagerank_networkx(graph, shape, damping=damping)
    if backend_norm == "scipy":
        return _pagerank_scipy(graph, shape, damping=damping)
    raise ValueError(f"Unknown pagerank backend {backend!r}; choose implicit, scipy, or networkx")
