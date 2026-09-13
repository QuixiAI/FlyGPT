"""Input/output neuron selection by degree only (README §5).

Degree is identical across RealFly and DegreePreservedFly, so the selected sets are
byte-identical across those two conditions. Ties break by node index, deterministically.
"""
from __future__ import annotations

import numpy as np

from .extract import EdgeGraph


def top_k_by(scores: np.ndarray, k: int) -> np.ndarray:
    order = np.lexsort((np.arange(len(scores)), -scores))  # score desc, index asc
    return np.sort(order[:k]).astype(np.int64)


def select_io(g: EdgeGraph, n_input: int, input_rule: str, n_output: int, output_rule: str):
    rules = {"top_out_degree": g.out_degree, "top_in_degree": g.in_degree}
    if input_rule not in rules or output_rule not in rules:
        raise ValueError(f"rules must be one of {list(rules)}")
    assert n_input <= g.n and n_output <= g.n, "graph smaller than requested interface"
    inp = top_k_by(rules[input_rule](), n_input)
    out = top_k_by(rules[output_rule](), n_output)
    info = {"n_input": int(n_input), "n_output": int(n_output), "input_rule": input_rule, "output_rule": output_rule,
            "overlap": int(len(np.intersect1d(inp, out)))}
    return inp, out, info
