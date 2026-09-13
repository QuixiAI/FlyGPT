"""FlyGPT: a language model whose recurrent core is the fruit-fly connectome."""
from __future__ import annotations

import torch.nn as nn

from .config import Config
from .graph import Graph, build_graph
from .model import FlyRNN
from .baselines import TinyRNN, TinyGPT


def build_model(cfg: Config, vocab_size: int) -> tuple[nn.Module, Graph | None]:
    """Construct whichever model the config names. Returns (model, graph-or-None)."""
    if cfg.model_type == "fly":
        graph = build_graph(cfg.graph)
        return FlyRNN(graph, vocab_size, cfg.model), graph
    if cfg.model_type == "rnn":
        return TinyRNN(vocab_size), None
    if cfg.model_type == "gpt":
        return TinyGPT(vocab_size, max_context=max(cfg.data.seq_len, 256)), None
    raise ValueError(f"unknown model_type {cfg.model_type!r}")


__all__ = ["Config", "Graph", "build_graph", "FlyRNN", "TinyRNN", "TinyGPT", "build_model"]
