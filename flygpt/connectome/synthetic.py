"""Synthetic connectome-shaped graphs for tests. Never used for a real run."""
from __future__ import annotations

import numpy as np

from .extract import EdgeGraph


def synthetic_edge_table(n: int = 400, density: float = 0.03, seed: int = 0, id_offset: int = 10**8):
    """Returns (src_ids, dst_ids, weight, regions) shaped like a neuPrint export with sparse body ids."""
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(10**9, n, replace=False)) + id_offset
    m = int(density * n * n)
    s, d = rng.integers(0, n, m), rng.integers(0, n, m)
    ring = np.arange(n)
    s, d = np.concatenate([s, ring, ring]), np.concatenate([d, np.roll(ring, -1), np.roll(ring, 1)])
    keep = s != d
    edges = np.unique(np.stack([s[keep], d[keep]], 1), axis=0)
    weight = rng.geometric(0.3, len(edges)).astype(np.float32)  # synapse counts, mostly small
    regions = {int(i): ("central_brain" if k < 0.8 * n else "vnc") for k, i in enumerate(ids)}
    return ids[edges[:, 0]], ids[edges[:, 1]], weight, regions


def synthetic_graph(n: int = 200, density: float = 0.05, seed: int = 0) -> EdgeGraph:
    src_ids, dst_ids, w, _ = synthetic_edge_table(n, density, seed)
    from .extract import from_edge_table
    return from_edge_table(src_ids, dst_ids, w)
