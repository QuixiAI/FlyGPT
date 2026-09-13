"""Tiny Shakespeare, char-level, fixed 90/10 split (plan.md §2).

`prepare_data.py` writes data/shakespeare/split.json: corpus sha256, split boundary,
vocab, and measured unigram/bigram reference losses on the actual split. That file is
committed; the corpus is not.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DatasetConfig

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
    """Contiguous token array with seeded random-window batching for truncated BPTT.

    The RNG is the *data order* stream: paired seeds share it across conditions.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int, batch_size: int, seed: int):
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.seq_len, self.batch_size = seq_len, batch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.tokens)

    def batch(self, device=None) -> tuple[torch.Tensor, torch.Tensor]:
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


def corpus_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def split_text(text: str, split: float) -> tuple[str, str, int]:
    boundary = int(len(text) * split)
    return text[:boundary], text[boundary:], boundary


def reference_losses(train_ids: np.ndarray, val_ids: np.ndarray, vocab_size: int, alpha: float = 1.0) -> dict:
    """Unigram and bigram cross-entropy (nats/char) on val, counts from train, add-alpha smoothing."""
    uni = np.bincount(train_ids, minlength=vocab_size).astype(np.float64) + alpha
    p_uni = uni / uni.sum()
    unigram = float(-np.log(p_uni[val_ids]).mean())

    bi = np.zeros((vocab_size, vocab_size), np.float64) + alpha
    np.add.at(bi, (train_ids[:-1], train_ids[1:]), 1)
    p_bi = bi / bi.sum(1, keepdims=True)
    bigram = float(-np.log(p_bi[val_ids[:-1], val_ids[1:]]).mean())
    return {"unigram_nats": unigram, "bigram_nats": bigram, "smoothing_alpha": alpha}


def prepare(cfg: DatasetConfig, out_path: str | Path = "data/shakespeare/split.json") -> dict:
    """Download, hash, split, measure references. Writes and returns split.json contents."""
    text = download_shakespeare(cfg.path).read_text()
    vocab = CharVocab.from_text(text)
    train_text, val_text, boundary = split_text(text, cfg.split)
    train_ids = np.array(vocab.encode(train_text))
    val_ids = np.array(vocab.encode(val_text))
    info = {
        "corpus_sha256": corpus_hash(text),
        "n_chars": len(text),
        "split": cfg.split,
        "split_boundary": boundary,
        "vocab": vocab.chars,
        "vocab_size": len(vocab),
        **reference_losses(train_ids, val_ids, len(vocab)),
    }
    Path(out_path).write_text(json.dumps(info, indent=2))
    return info


def load_split(cfg: DatasetConfig, split_json: str | Path = "data/shakespeare/split.json"):
    """Returns (vocab, train_ids, val_ids). Verifies the corpus hash against split.json when present."""
    text = download_shakespeare(cfg.path).read_text()
    sj = Path(split_json)
    if sj.exists() and cfg.max_chars is None:
        info = json.loads(sj.read_text())
        assert info["corpus_sha256"] == corpus_hash(text), "corpus changed; re-run prepare_data.py"
        vocab = CharVocab(info["vocab"])
        boundary = info["split_boundary"]
    else:
        vocab = CharVocab.from_text(text)
        boundary = None
    if cfg.max_chars:
        text = text[: cfg.max_chars]
        boundary = None
    if boundary is None:
        _, _, boundary = split_text(text, cfg.split)
    ids = np.array(vocab.encode(text), dtype=np.int64)
    return vocab, ids[:boundary], ids[boundary:]


def make_streams(cfg: DatasetConfig, seq_len: int, batch_size: int, seed: int):
    vocab, train_ids, val_ids = load_split(cfg)
    if len(val_ids) < seq_len + 2:  # tiny dev corpora
        val_ids = train_ids[-(seq_len + 2):]
    return (vocab,
            TokenStream(train_ids, seq_len, batch_size, seed),
            TokenStream(val_ids, seq_len, batch_size, 10_000 + seed))
