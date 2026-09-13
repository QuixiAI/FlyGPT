"""Controls (plan.md §4). Every control uses exactly the real subgraph's neurons.

  degree_preserving  — directed double-edge swaps; in/out degree of every neuron preserved  (primary)
  uniform_random     — same N and E, endpoints fully random                                   (secondary)

Synapse weights: the original multiset of synapse counts is randomly reassigned to the
rewired edges (§4.2), so magnitudes are matched and only "who connects to whom" differs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import EdgeGraph


@dataclass
class RewireResult:
    graph: EdgeGraph
    seed: int
    accepted_swaps: int
    survival_history: list[float]     # fraction of original edges present, sampled once per chunk

    @property
    def survival(self) -> float:
        return self.survival_history[-1]


def _edge_key(src: np.ndarray, dst: np.ndarray, n: int) -> np.ndarray:
    return src.astype(np.int64) * n + dst.astype(np.int64)


def degree_preserving_rewire(g: EdgeGraph, seed: int, min_swap_factor: int = 10, chunk_factor: float = 1.0,
                             plateau_tol: float = 1e-3, max_swap_factor: int = 50) -> RewireResult:
    """Directed double-edge swaps: (a->b, c->d) becomes (a->d, c->b).

    Rejects self-loops and duplicates. Runs at least `min_swap_factor * E` accepted swaps, then keeps
    swapping in chunks of `chunk_factor * E` until the original-edge survival fraction stops decreasing
    by more than `plateau_tol` per chunk (or `max_swap_factor * E` is reached).
    """
    rng = np.random.default_rng(seed)
    n, E = g.n, g.n_edges
    src, dst = g.src.copy(), g.dst.copy()
    present = set(_edge_key(src, dst, n).tolist())
    original = set(present)
    accepted = 0
    history: list[float] = []
    chunk = max(int(chunk_factor * E), 1)
    min_swaps, max_swaps = min_swap_factor * E, max_swap_factor * E

    def survival() -> float:
        return len(original & present) / E

    while True:
        done_in_chunk = 0
        while done_in_chunk < chunk:
            i, j = rng.integers(0, E, 2)
            if i == j:
                continue
            a, b, c, d = src[i], dst[i], src[j], dst[j]
            if a == d or c == b:
                continue
            k1, k2 = a * n + d, c * n + b
            if k1 in present or k2 in present:
                continue
            present.discard(a * n + b)
            present.discard(c * n + d)
            present.add(k1)
            present.add(k2)
            dst[i], dst[j] = d, b
            accepted += 1
            done_in_chunk += 1
        history.append(survival())
        if accepted >= max_swaps:
            break
        if accepted >= min_swaps and len(history) >= 2 and history[-2] - history[-1] < plateau_tol:
            break

    weight = rng.permutation(g.weight)  # §4.2: same multiset of synapse counts, reassigned
    return RewireResult(EdgeGraph(n, src, dst, weight, g.ids.copy()), seed, accepted, history)


def plateau_agrees(a: RewireResult, b: RewireResult, tol: float | None = None) -> bool:
    """Two independent shuffles should land on the same survival fraction (plan.md §4).

    Default tolerance: 4 binomial standard deviations of the survival fraction at this edge count,
    so a 700-edge test graph and a 500k-edge real graph are judged on the same footing."""
    if tol is None:
        E = a.graph.n_edges
        p = max(min((a.survival + b.survival) / 2, 0.5), 1e-3)
        tol = 4 * (p * (1 - p) / E) ** 0.5
    return abs(a.survival - b.survival) <= tol


def uniform_random_rewire(g: EdgeGraph, seed: int) -> EdgeGraph:
    rng = np.random.default_rng(seed)
    n, E = g.n, g.n_edges
    seen: set[int] = set()
    src_out, dst_out = [], []
    while len(src_out) < E:
        need = E - len(src_out)
        s = rng.integers(0, n, need * 2)
        d = rng.integers(0, n, need * 2)
        for a, b in zip(s.tolist(), d.tolist()):
            if a == b or (a * n + b) in seen:
                continue
            seen.add(a * n + b)
            src_out.append(a)
            dst_out.append(b)
            if len(src_out) == E:
                break
    weight = rng.permutation(g.weight)
    return EdgeGraph(n, np.array(src_out, np.int64), np.array(dst_out, np.int64), weight, g.ids.copy())


def build_control(g: EdgeGraph, name: str, seed: int) -> tuple[EdgeGraph, dict]:
    """Dispatch by control name. Returns (graph, provenance dict for the stats file)."""
    if name == "real":
        return g, {"control": "real"}
    if name == "degree_preserving":
        r1 = degree_preserving_rewire(g, seed)
        r2 = degree_preserving_rewire(g, seed + 1000)
        info = {"control": "degree_preserving", "seed": seed, "accepted_swaps": r1.accepted_swaps,
                "edge_survival": r1.survival, "survival_history": r1.survival_history,
                "second_shuffle_survival": r2.survival, "plateau_agrees": plateau_agrees(r1, r2)}
        return r1.graph, info
    if name == "uniform_random":
        return uniform_random_rewire(g, seed), {"control": "uniform_random", "seed": seed}
    raise ValueError(f"unknown control {name!r}")
