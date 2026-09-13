"""Parameter-matched engineered baselines (plan.md §11): tanh RNN, GRU, tiny Transformer.

`match_hidden` picks the hidden size whose trainable parameter count is closest to a target,
so each baseline can be matched to a FlyGPT config's total.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GenerateMixin:
    max_context: int | None = None

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int, temperature: float = 1.0, top_k: int | None = None):
        was_training = self.training
        self.eval()
        out = prompt
        for _ in range(max_new):
            ctx = out[:, -self.max_context:] if self.max_context else out
            lg, _ = self(ctx)
            lg = lg[0, -1] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(lg, min(top_k, lg.numel()))
                lg[lg < v[-1]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1).view(1, 1)
            out = torch.cat([out, nxt], 1)
        self.train(was_training)
        return out

    def parameter_counts(self) -> dict:
        rec = sum(p.numel() for p in self.recurrent_parameters())
        ad = sum(p.numel() for p in self.adapter_parameters())
        return {"recurrent": rec, "input": 0, "output": ad, "total": rec + ad, "adapter_to_recurrent_ratio": ad / max(rec, 1)}


class TinyRNN(nn.Module, _GenerateMixin):
    kind = "rnn"

    def __init__(self, vocab_size: int, hidden: int = 256, embed_dim: int = 32, cell: str = "rnn"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = (nn.GRU(embed_dim, hidden, batch_first=True) if cell == "gru"
                    else nn.RNN(embed_dim, hidden, nonlinearity="tanh", batch_first=True))
        self.readout = nn.Linear(hidden, vocab_size)

    def recurrent_parameters(self):
        return list(self.rnn.parameters())

    def adapter_parameters(self):
        return list(self.embed.parameters()) + list(self.readout.parameters())

    def forward(self, tokens, state=None):
        h, state = self.rnn(self.embed(tokens), state)
        return self.readout(h), state


class TinyGPT(nn.Module, _GenerateMixin):
    kind = "transformer"

    def __init__(self, vocab_size: int, d_model: int = 128, n_layer: int = 4, n_head: int = 4, max_context: int = 256):
        super().__init__()
        self.max_context = max_context
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_context, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, vocab_size)

    def recurrent_parameters(self):
        return list(self.blocks.parameters())

    def adapter_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("blocks.")]

    def forward(self, tokens, state=None):
        B, T = tokens.shape
        x = self.tok(tokens) + self.pos(torch.arange(T, device=tokens.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens.device)
        return self.readout(self.ln(self.blocks(x, mask=mask, is_causal=True))), None


def build_baseline(kind: str, vocab_size: int, hidden: int, max_context: int = 256) -> nn.Module:
    if kind in ("rnn", "gru"):
        return TinyRNN(vocab_size, hidden=hidden, cell=kind)
    if kind == "transformer":
        return TinyGPT(vocab_size, d_model=hidden, max_context=max_context)
    raise ValueError(kind)


def match_hidden(kind: str, vocab_size: int, target_params: int, lo: int = 8, hi: int = 4096) -> int:
    """Hidden size whose total parameter count is closest to `target_params` (transformer: multiple of 4 heads)."""
    step = 4 if kind == "transformer" else 1
    best, best_gap = lo, float("inf")
    a, b = lo, hi
    while b - a > step:  # param count is monotone in hidden size
        mid = ((a + b) // 2) // step * step
        n = build_baseline(kind, vocab_size, mid).parameter_counts()["total"]
        gap = abs(n - target_params)
        if gap < best_gap:
            best, best_gap = mid, gap
        if n < target_params:
            a = mid
        else:
            b = mid
    return best
