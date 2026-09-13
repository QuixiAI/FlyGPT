"""Scoreboard baselines: a tiny vanilla RNN and a tiny GPT.

Both expose the same interface as FlyRNN: forward(tokens, state) -> (logits, state), generate(...).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GenerateMixin:
    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int, temperature: float = 1.0, top_k: int | None = None):
        self.eval()
        out = prompt
        for _ in range(max_new):
            lg, _ = self(out[:, -self.max_context:] if hasattr(self, "max_context") else out)
            lg = lg[0, -1] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(lg, min(top_k, lg.numel()))
                lg[lg < v[-1]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1).view(1, 1)
            out = torch.cat([out, nxt], 1)
        return out


class TinyRNN(nn.Module, _GenerateMixin):
    def __init__(self, vocab_size: int, hidden: int = 256, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden, nonlinearity="tanh", batch_first=True)
        self.readout = nn.Linear(hidden, vocab_size)

    def recurrent_parameters(self):
        return list(self.rnn.parameters())

    def adapter_parameters(self):
        return list(self.embed.parameters()) + list(self.readout.parameters())

    def forward(self, tokens, state=None):
        h, state = self.rnn(self.embed(tokens), state)
        return self.readout(h), state


class TinyGPT(nn.Module, _GenerateMixin):
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
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.readout(self.ln(x)), None
