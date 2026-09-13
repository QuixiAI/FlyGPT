"""Datasets: Tiny Shakespeare (char-level) plus tiny synthetic tasks for step 1."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DataConfig

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class CharVocab:
    chars: list[str]

    @classmethod
    def from_text(cls, text: str) -> "CharVocab":
        return cls(sorted(set(text)))

    def __len__(self):
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        m = {c: i for i, c in enumerate(self.chars)}
        return [m[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


class TokenStream:
    """Contiguous token array with random-window batching for truncated BPTT."""

    def __init__(self, tokens: np.ndarray, seq_len: int, batch_size: int, seed: int = 0):
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.seq_len, self.batch_size = seq_len, batch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tokens)

    def batch(self, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, y) each [B, T]; y is x shifted by one."""
        hi = len(self.tokens) - self.seq_len - 1
        starts = self.rng.integers(0, hi, self.batch_size)
        x = torch.stack([self.tokens[s:s + self.seq_len] for s in starts])
        y = torch.stack([self.tokens[s + 1:s + self.seq_len + 1] for s in starts])
        return x.to(device), y.to(device)


def download_shakespeare(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        import requests
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(requests.get(SHAKESPEARE_URL, timeout=60).text)
    return path


def synthetic_text(task: str, n_chars: int = 20_000, seed: int = 0) -> str:
    """Toy corpora for checking that a recurrent graph can learn at all."""
    rng = np.random.default_rng(seed)
    if task == "repeat":
        motif = "abcdefgh"
        return (motif * (n_chars // len(motif) + 1))[:n_chars]
    if task == "delayed_copy":
        # random symbol stream where each char reappears exactly `delay` positions later
        delay = 8
        base = list("xyzw")
        out = [base[i] for i in rng.integers(0, len(base), delay)]
        while len(out) < n_chars:
            out.append(out[-delay])
        return "".join(out)
    raise ValueError(f"unknown synthetic task {task!r}")


def load_text(cfg: DataConfig) -> str:
    if cfg.task == "shakespeare":
        text = download_shakespeare(cfg.path).read_text()
    else:
        text = synthetic_text(cfg.task)
    if cfg.max_chars:
        text = text[: cfg.max_chars]
    return text


def make_streams(cfg: DataConfig, seed: int = 0) -> tuple[CharVocab, TokenStream, TokenStream]:
    text = load_text(cfg)
    vocab = CharVocab.from_text(text)
    ids = np.array(vocab.encode(text), dtype=np.int64)
    n_val = max(int(len(ids) * cfg.val_fraction), cfg.seq_len + 2)
    train, val = ids[:-n_val], ids[-n_val:]
    return (vocab,
            TokenStream(train, cfg.seq_len, cfg.batch_size, seed),
            TokenStream(val, cfg.seq_len, cfg.batch_size, seed + 1))
