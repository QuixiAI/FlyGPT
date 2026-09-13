"""FlyRNN: embedding -> input neurons -> scatter-add recurrence over the connectome -> readout.

    messages = state[src] * edge_weight
    incoming = scatter_add(messages, dst)
    new_state = tanh(incoming + input + bias)

One scalar hidden state per neuron. One trainable weight per real edge.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .graph import Graph


class FlyRNN(nn.Module):
    def __init__(self, graph: Graph, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.n = graph.n
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.register_buffer("src", torch.as_tensor(graph.src, dtype=torch.long))
        self.register_buffer("dst", torch.as_tensor(graph.dst, dtype=torch.long))
        self.register_buffer("input_nodes", torch.as_tensor(graph.input_nodes, dtype=torch.long))
        self.register_buffer("output_nodes", torch.as_tensor(graph.output_nodes, dtype=torch.long))

        # Recurrent core: one weight per edge, one bias per neuron.
        in_deg = torch.bincount(self.dst, minlength=self.n).clamp(min=1).float()
        self.edge_weight = nn.Parameter(torch.randn(self.src.numel()) * cfg.init_scale / in_deg[self.dst].sqrt())
        self.bias = nn.Parameter(torch.zeros(self.n))

        # Adapters: kept tiny relative to the graph.
        self.embed = nn.Embedding(vocab_size, cfg.embed_dim)
        self.inp = nn.Linear(cfg.embed_dim, len(graph.input_nodes))
        self.readout = nn.Linear(len(graph.output_nodes), vocab_size)

    def recurrent_parameters(self):
        return [self.edge_weight, self.bias]

    def adapter_parameters(self):
        return list(self.embed.parameters()) + list(self.inp.parameters()) + list(self.readout.parameters())

    def init_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.n, device=device or self.edge_weight.device)

    def step(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """One character in: x [B] token ids, state [B, n] -> new state [B, n]."""
        drive = torch.zeros_like(state)
        drive[:, self.input_nodes] = self.inp(self.embed(x))
        for _ in range(self.cfg.microsteps):
            messages = state[:, self.src] * self.edge_weight            # [B, E]
            incoming = torch.zeros_like(state).index_add_(1, self.dst, messages)
            act = torch.tanh(incoming + drive + self.bias)
            state = (1 - self.cfg.leak) * state + self.cfg.leak * act
        return state

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.readout(state[:, self.output_nodes])

    def forward(self, tokens: torch.Tensor, state: torch.Tensor | None = None):
        """tokens [B, T] -> logits [B, T, V], final state."""
        B, T = tokens.shape
        if state is None:
            state = self.init_state(B, tokens.device)
        outs = []
        for t in range(T):
            state = self.step(state, tokens[:, t])
            outs.append(self.logits(state))
        return torch.stack(outs, 1), state

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int, temperature: float = 1.0,
                 top_k: int | None = None, return_states: bool = False):
        """prompt [1, T] -> generated ids [1, T + max_new]. Optionally returns per-step states."""
        self.eval()
        state = self.init_state(1, prompt.device)
        states = []
        for t in range(prompt.shape[1]):
            state = self.step(state, prompt[:, t])
            if return_states:
                states.append(state.clone())
        out = prompt
        for _ in range(max_new):
            lg = self.logits(state)[0] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(lg, min(top_k, lg.numel()))
                lg[lg < v[-1]] = -float("inf")
            nxt = torch.multinomial(F.softmax(lg, -1), 1).view(1, 1)
            out = torch.cat([out, nxt], 1)
            state = self.step(state, nxt[:, 0])
            if return_states:
                states.append(state.clone())
        return (out, torch.cat(states, 0)) if return_states else out


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
