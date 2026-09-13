"""Load a checkpoint back into a model + vocab."""
from __future__ import annotations

import torch

from . import Config, build_model
from .data import CharVocab


def load_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    vocab = CharVocab(ckpt["vocab"])
    model, graph = build_model(cfg, len(vocab))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, graph, vocab, cfg, ckpt
