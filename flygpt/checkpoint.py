"""Save/load a trained model together with everything needed to rebuild it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import Config
from .data import CharVocab


def save_checkpoint(path, model, cfg: Config, vocab: CharVocab, condition: str, seed: int, step: int, val_loss: float,
                    graph_meta: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.to_dict(), "vocab": vocab.chars, "condition": condition, "seed": seed,
                "step": step, "val_loss": val_loss, "graph_meta": graph_meta or {},
                "model": model.state_dict()}, path)


def load_checkpoint(path: str, device="cpu"):
    """Rebuilds the model from the graph artifacts referenced in the checkpoint's config."""
    from . import build_model
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    vocab = CharVocab(ckpt["vocab"])
    model, meta = build_model(cfg, ckpt["condition"], ckpt["seed"], len(vocab))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, meta, vocab, cfg, ckpt
