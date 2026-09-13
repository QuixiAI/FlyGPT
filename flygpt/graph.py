"""Graph loading and subgraph selection.

A Graph is just: n nodes, directed edges (src, dst), and chosen input/output nodes.
Node ids are contiguous 0..n-1 after subgraph extraction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GraphConfig


@dataclass
class Graph:
    n: int
    src: np.ndarray          # int64 [E]
    dst: np.ndarray          # int64 [E]
    input_nodes: np.ndarray  # int64 [K_in]
    output_nodes: np.ndarray # int64 [K_out]
    original_ids: np.ndarray | None = None  # fly body ids, if from the connectome

    @property
    def n_edges(self) -> int:
        return int(self.src.shape[0])


def synthetic_graph(n: int, density: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random directed graph without self-loops. Guaranteed weakly connected via a ring."""
    rng = np.random.default_rng(seed)
    n_rand = int(density * n * n)
    src = rng.integers(0, n, n_rand)
    dst = rng.integers(0, n, n_rand)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    ring = np.arange(n)
    src = np.concatenate([src, ring])
    dst = np.concatenate([dst, np.roll(ring, -1)])
    edges = np.unique(np.stack([src, dst], 1), axis=0)
    return edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)


def load_fly_edges(edges_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (src_ids, dst_ids, weight) from the parquet produced by data/fly/build_edges.py."""
    import pandas as pd

    df = pd.read_parquet(edges_path)
    return df["src"].to_numpy(np.int64), df["dst"].to_numpy(np.int64), df["weight"].to_numpy(np.float32)


def largest_weak_component(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Node mask for the largest weakly connected component (union-find)."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(src.tolist(), dst.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    biggest = np.bincount(roots).argmax()
    return roots == biggest


def select_subgraph(src: np.ndarray, dst: np.ndarray, n_target: int, seed: int,
                    node_ids: np.ndarray | None = None):
    """Pick ~n_target nodes forming a connected subgraph.

    Strategy: rank nodes by total degree, take the top n_target, then keep the
    largest weakly connected component. Relabel to 0..n-1.
    """
    all_ids = np.unique(np.concatenate([src, dst])) if node_ids is None else node_ids
    idx = {v: i for i, v in enumerate(all_ids.tolist())}
    s = np.array([idx[v] for v in src.tolist()], dtype=np.int64)
    d = np.array([idx[v] for v in dst.tolist()], dtype=np.int64)
    n_all = len(all_ids)
    deg = np.bincount(s, minlength=n_all) + np.bincount(d, minlength=n_all)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n_all), -deg))  # degree desc, random tiebreak
    chosen = np.zeros(n_all, bool)
    chosen[order[:n_target]] = True
    keep_e = chosen[s] & chosen[d]
    s, d = s[keep_e], d[keep_e]
    comp = largest_weak_component(n_all, s, d) & chosen
    keep_e = comp[s] & comp[d]
    s, d = s[keep_e], d[keep_e]
    new_id = -np.ones(n_all, np.int64)
    new_id[comp] = np.arange(comp.sum())
    return int(comp.sum()), new_id[s], new_id[d], all_ids[comp]


def choose_io_nodes(n: int, src: np.ndarray, dst: np.ndarray, n_input: int, n_output: int,
                    rule: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic input/output node selection. Input and output sets are disjoint."""
    rng = np.random.default_rng(seed + 1)
    if rule == "random":
        perm = rng.permutation(n)
    elif rule == "high_degree":
        deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
        perm = np.lexsort((rng.random(n), -deg))
    else:
        raise ValueError(f"unknown io_rule {rule!r}")
    assert n_input + n_output <= n, "graph too small for requested I/O sizes"
    return perm[:n_input].astype(np.int64), perm[n_input:n_input + n_output].astype(np.int64)


def build_graph(cfg: GraphConfig) -> Graph:
    if cfg.source == "synthetic":
        src, dst = synthetic_graph(cfg.n_neurons, cfg.synthetic_density, cfg.seed)
        n, original = cfg.n_neurons, None
    elif cfg.source == "fly":
        raw_src, raw_dst, _ = load_fly_edges(cfg.edges_path)
        n, src, dst, original = select_subgraph(raw_src, raw_dst, cfg.n_neurons, cfg.seed)
    else:
        raise ValueError(f"unknown graph source {cfg.source!r}")

    if cfg.scramble:
        from .scramble import scramble_edges
        src, dst = scramble_edges(n, src, dst, cfg.seed)

    inp, out = choose_io_nodes(n, src, dst, cfg.n_input, cfg.n_output, cfg.io_rule, cfg.seed)
    return Graph(n=n, src=src, dst=dst, input_nodes=inp, output_nodes=out, original_ids=original)
