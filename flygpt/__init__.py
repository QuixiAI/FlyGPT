"""FlyGPT: a language model whose recurrent core is a real subgraph of the fruit-fly connectome."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .connectome import EdgeGraph, load_subgraph, select_io
from .model import FlyRNN, make_frozen_fly
from .baselines import build_baseline, match_hidden

CONDITIONS = ("real", "degree_preserving", "uniform_random", "frozen")


def condition_dir(cfg: Config, condition: str, seed: int) -> Path:
    """Where build_graph.py wrote this condition's edges. `real` and `frozen` share the real graph."""
    base = cfg.graph_dir
    if condition in ("real", "frozen"):
        return base / "real"
    return base / f"{condition}_seed{seed}"


def load_condition(cfg: Config, condition: str, seed: int):
    """Returns (graph, inputs, outputs, meta) for one condition, from artifacts on disk."""
    d = condition_dir(cfg, condition, seed)
    g = load_subgraph(d / "edges.pt")
    inp, out, io_info = select_io(g, cfg.interface.input_nodes, cfg.interface.input_rule,
                                  cfg.interface.output_nodes, cfg.interface.output_rule)
    meta = {"condition": condition, "dir": str(d), "io": io_info}
    diag = d / "diagnostics.json"
    if diag.exists():
        meta["diagnostics"] = json.loads(diag.read_text())
    return g, inp, out, meta


def build_model(cfg: Config, condition: str, seed: int, vocab_size: int):
    """Paired seeds (§12): adapters init from torch global RNG seeded with `seed`; edge values from
    a dedicated generator seeded with `seed`. Both are identical across conditions for the same k."""
    if condition in ("rnn", "gru", "transformer"):
        torch.manual_seed(seed)
        hidden = match_hidden(condition, vocab_size, _fly_param_target(cfg, seed, vocab_size))
        model = build_baseline(condition, vocab_size, hidden, max_context=max(cfg.sequence.context, 256))
        return model, {"condition": condition, "hidden": hidden, "params": model.parameter_counts()}
    g, inp, out, meta = load_condition(cfg, condition, seed)
    torch.manual_seed(seed)
    if condition == "frozen":
        model = make_frozen_fly(g, inp, out, vocab_size, cfg.model, cfg.sequence, seed)
    else:
        edge_gen = torch.Generator().manual_seed(seed)
        model = FlyRNN(g, inp, out, vocab_size, cfg.model, cfg.sequence, edge_generator=edge_gen)
    meta["params"] = model.parameter_counts()
    return model, meta


def _fly_param_target(cfg: Config, seed: int, vocab_size: int) -> int:
    g, inp, out, _ = load_condition(cfg, "real", seed)
    return FlyRNN(g, inp, out, vocab_size, cfg.model, cfg.sequence).parameter_counts()["total"]


__all__ = ["Config", "EdgeGraph", "FlyRNN", "build_model", "load_condition", "condition_dir", "CONDITIONS"]
