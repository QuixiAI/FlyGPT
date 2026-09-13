"""Deterministic dense-core subgraph extraction (plan.md §3).

Random sampling is prohibited. The procedure is:
  A. filter edges by synapse count and region
  B. largest strongly connected component
  C. directed-core pruning: largest (in>=k, out>=k) core with >= target nodes
  D. deterministic trim by weighted degree, recompute SCC, re-trim with slack if the SCC falls short
  E. save exact ids, edges, config, hash, stats
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..config import GraphConfig


@dataclass
class EdgeGraph:
    """Directed graph on nodes 0..n-1. `ids` maps local index -> original body id."""
    n: int
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    ids: np.ndarray

    @property
    def n_edges(self) -> int:
        return int(len(self.src))

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.dst, minlength=self.n)

    def out_degree(self) -> np.ndarray:
        return np.bincount(self.src, minlength=self.n)

    def weighted_degree(self) -> np.ndarray:
        return (np.bincount(self.dst, weights=self.weight, minlength=self.n)
                + np.bincount(self.src, weights=self.weight, minlength=self.n))

    def induced(self, keep: np.ndarray) -> "EdgeGraph":
        """Subgraph on boolean mask `keep`, relabelled to 0..k-1."""
        new_id = -np.ones(self.n, np.int64)
        new_id[keep] = np.arange(int(keep.sum()))
        m = keep[self.src] & keep[self.dst]
        return EdgeGraph(int(keep.sum()), new_id[self.src[m]], new_id[self.dst[m]], self.weight[m], self.ids[keep])

    def csr(self):
        return coo_matrix((np.ones(self.n_edges), (self.src, self.dst)), shape=(self.n, self.n)).tocsr()

    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(np.sort(self.ids).tobytes())
        order = np.lexsort((self.dst, self.src))
        h.update(self.ids[self.src[order]].tobytes())
        h.update(self.ids[self.dst[order]].tobytes())
        return h.hexdigest()


def from_edge_table(src_ids: np.ndarray, dst_ids: np.ndarray, weight: np.ndarray,
                    candidate_ids: np.ndarray | None = None) -> EdgeGraph:
    """Build an EdgeGraph from body-id columns, optionally restricted to `candidate_ids`."""
    if candidate_ids is not None:
        cand = np.unique(candidate_ids)
        m = np.isin(src_ids, cand) & np.isin(dst_ids, cand)
        src_ids, dst_ids, weight = src_ids[m], dst_ids[m], weight[m]
    ids = np.unique(np.concatenate([src_ids, dst_ids]))
    src = np.searchsorted(ids, src_ids)
    dst = np.searchsorted(ids, dst_ids)
    m = src != dst
    return EdgeGraph(len(ids), src[m].astype(np.int64), dst[m].astype(np.int64), weight[m].astype(np.float32), ids)


def largest_scc(g: EdgeGraph) -> np.ndarray:
    """Boolean mask of the largest strongly connected component."""
    if g.n == 0:
        return np.zeros(0, bool)
    _, labels = connected_components(g.csr(), directed=True, connection="strong")
    return labels == np.bincount(labels).argmax()


def directed_core(g: EdgeGraph, k: int) -> np.ndarray:
    """Mask of nodes surviving iterative removal of nodes with in<k or out<k."""
    keep = np.ones(g.n, bool)
    while True:
        m = keep[g.src] & keep[g.dst]
        indeg = np.bincount(g.dst[m], minlength=g.n)
        outdeg = np.bincount(g.src[m], minlength=g.n)
        drop = keep & ((indeg < k) | (outdeg < k))
        if not drop.any():
            return keep
        keep &= ~drop


def largest_k_with_core(g: EdgeGraph, target: int, k_max: int | None = None) -> int:
    """Largest k such that the (k,k)-directed core has at least `target` nodes (0 if none)."""
    k_max = k_max or int(max(g.in_degree().max(), g.out_degree().max()))
    lo, hi = 0, k_max
    while lo < hi:  # core size is monotone non-increasing in k
        mid = (lo + hi + 1) // 2
        if directed_core(g, mid).sum() >= target:
            lo = mid
        else:
            hi = mid - 1
    return lo


def extract_dense_core(g: EdgeGraph, target: int, min_synapses: int, retain_scc: bool = True) -> tuple[EdgeGraph, dict]:
    """Steps A-D. Returns the final subgraph and a log of the decisions taken."""
    log = {"candidate_pool": int(g.n), "candidate_edges": int(g.n_edges), "min_synapses": min_synapses, "target_neurons": target}
    m = g.weight >= min_synapses
    g = EdgeGraph(g.n, g.src[m], g.dst[m], g.weight[m], g.ids)
    log["edges_after_threshold"] = int(g.n_edges)

    if retain_scc:
        g = g.induced(largest_scc(g))
    log["scc_after_threshold"] = int(g.n)
    if g.n <= target:
        log["note"] = "SCC no larger than target; no pruning or trimming applied"
        log.update(core_k=0, final_neurons=int(g.n), final_edges=int(g.n_edges))
        return g, log

    k = largest_k_with_core(g, target)
    core = g.induced(directed_core(g, k))
    log["core_size_before_trim"] = int(core.n)
    # D. Trimming to exactly `target` by weighted degree can leave an SCC a few nodes short. Lowering k
    # cannot fix that (the core is already >= target), so instead trim to target + deficit and retry.
    # Result: the smallest SCC >= target reachable this way. Logged in trim_iterations / final_neurons.
    trimmed, m, iters = core, target, 0
    while core.n > target:
        rank = np.argsort(-core.weighted_degree(), kind="stable")
        keep = np.zeros(core.n, bool)
        keep[rank[:m]] = True
        trimmed = core.induced(keep)
        if retain_scc:
            trimmed = trimmed.induced(largest_scc(trimmed))
        iters += 1
        if trimmed.n >= target or m >= core.n or iters > 50:
            break
        m += target - trimmed.n
    core = trimmed
    log["trim_iterations"] = iters
    log.update(core_k=int(k), final_neurons=int(core.n), final_edges=int(core.n_edges))
    return core, log


def graph_stats(g: EdgeGraph, regions: dict[int, str] | None = None) -> dict:
    indeg, outdeg = g.in_degree(), g.out_degree()
    pairs = set(zip(g.src.tolist(), g.dst.tolist()))
    reciprocal = sum(1 for a, b in pairs if a < b and (b, a) in pairs)
    stats = {
        "n_neurons": int(g.n),
        "n_edges": int(g.n_edges),
        "synaptic_contacts": float(g.weight.sum()),
        "largest_scc_fraction": float(largest_scc(g).mean()) if g.n else 0.0,
        "reciprocal_pairs": int(reciprocal),
        "in_degree": {"mean": float(indeg.mean()), "min": int(indeg.min()), "max": int(indeg.max())},
        "out_degree": {"mean": float(outdeg.mean()), "min": int(outdeg.min()), "max": int(outdeg.max())},
        "density": float(g.n_edges / max(g.n * (g.n - 1), 1)),
        "hash": g.hash(),
    }
    if regions is not None:
        counts: dict[str, int] = {}
        for i in g.ids.tolist():
            r = regions.get(i, "unknown")
            counts[r] = counts.get(r, 0) + 1
        stats["region_breakdown"] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return stats


def save_subgraph(g: EdgeGraph, cfg: GraphConfig, out_dir: str | Path, extra_stats: dict | None = None) -> dict:
    """Step E: subgraph_node_ids.txt, subgraph_edges.pt, subgraph_config.yaml, subgraph_hash.txt, subgraph_stats.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "subgraph_node_ids.txt", g.ids, fmt="%d")
    torch.save({"src": torch.as_tensor(g.src), "dst": torch.as_tensor(g.dst),
                "weight": torch.as_tensor(g.weight), "ids": torch.as_tensor(g.ids), "n": g.n}, out / "subgraph_edges.pt")
    (out / "subgraph_config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    (out / "subgraph_hash.txt").write_text(g.hash() + "\n")
    stats = {**(extra_stats or {}), **graph_stats(g)}
    (out / "subgraph_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_subgraph(path: str | Path) -> EdgeGraph:
    d = torch.load(Path(path), weights_only=False)
    return EdgeGraph(int(d["n"]), d["src"].numpy(), d["dst"].numpy(), d["weight"].numpy(), d["ids"].numpy())
