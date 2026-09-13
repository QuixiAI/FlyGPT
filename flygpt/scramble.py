"""ScrambledFly control: same node count, same edge count, random wiring.

V0 uses the simplest possible scramble (uniform random endpoints, no self-loops,
no duplicate edges). Degree-preserving rewiring is a later experiment.
"""
from __future__ import annotations

import numpy as np


def scramble_edges(n: int, src: np.ndarray, dst: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 12345)
    n_edges = src.shape[0]
    seen = set()
    out_src, out_dst = [], []
    while len(out_src) < n_edges:
        need = n_edges - len(out_src)
        s = rng.integers(0, n, need * 2)
        d = rng.integers(0, n, need * 2)
        for a, b in zip(s.tolist(), d.tolist()):
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            out_src.append(a)
            out_dst.append(b)
            if len(out_src) == n_edges:
                break
    return np.array(out_src, np.int64), np.array(out_dst, np.int64)
