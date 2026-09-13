"""Per-condition diagnostics and the path-length gate (plan.md §4.1, §6).

Run on every graph (real and each control) before training. The gate is go/no-go:
a failing graph is fixed at the subgraph/interface level, never by tuning microsteps.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import shortest_path

from ..config import GateConfig
from .extract import EdgeGraph, largest_scc


def io_path_lengths(g: EdgeGraph, inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """For each output node, the shortest directed hop count from any input node (inf if unreachable)."""
    dist = shortest_path(g.csr(), directed=True, unweighted=True, indices=inputs)  # [K_in, N]
    return dist[:, outputs].min(0)


def diagnose(g: EdgeGraph, inputs: np.ndarray, outputs: np.ndarray) -> dict:
    pairs = set(zip(g.src.tolist(), g.dst.tolist()))
    reciprocal = sum(1 for a, b in pairs if a < b and (b, a) in pairs)
    d = io_path_lengths(g, inputs, outputs)
    finite = d[np.isfinite(d)]
    return {
        "n_neurons": int(g.n),
        "n_edges": int(g.n_edges),
        "largest_scc_fraction": float(largest_scc(g).mean()),
        "reciprocal_pairs": int(reciprocal),
        "io_reachable_fraction": float(np.isfinite(d).mean()),
        "io_path_median": float(np.median(finite)) if len(finite) else None,
        "io_path_p90": float(np.percentile(finite, 90)) if len(finite) else None,
        "io_path_max": float(finite.max()) if len(finite) else None,
    }


def path_gate(diag: dict, microsteps: int, gate: GateConfig) -> dict:
    """plan.md §6: reachable fraction high, p90 path fits within `max_p90_chars` characters at `microsteps`."""
    max_hops = microsteps * gate.max_p90_chars
    ok_reach = diag["io_reachable_fraction"] >= gate.min_reachable_fraction
    ok_p90 = diag["io_path_p90"] is not None and diag["io_path_p90"] <= max_hops
    return {"microsteps": microsteps, "max_p90_hops": max_hops, "min_reachable_fraction": gate.min_reachable_fraction,
            "reachable_ok": bool(ok_reach), "p90_ok": bool(ok_p90), "passed": bool(ok_reach and ok_p90)}
