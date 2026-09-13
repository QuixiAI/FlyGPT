"""FlyRNN: batched sparse recurrence over the connectome (plan.md §7, §8).

    proposal_i = tanh( normalized_recurrent_input_i + external_input_i + bias_i )
    h_i_new    = (1 - leak_i) * h_i + leak_i * proposal_i

One scalar state per neuron. One trainable value per real edge. The recurrent matrix is
stored with rows = destination, columns = source, so `incoming = sparse.mm(W, state.T).T`.
`[B, T, E]` messages are never materialized. The sparse op runs in fp32.

`dense_reference_step` is the slow dense path used only by the unit test.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, SequenceConfig
from .connectome.extract import EdgeGraph


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


class FlyRNN(nn.Module):
    def __init__(self, graph: EdgeGraph, inputs: np.ndarray, outputs: np.ndarray, vocab_size: int,
                 mcfg: ModelConfig, scfg: SequenceConfig, edge_generator: torch.Generator | None = None,
                 frozen: bool = False):
        super().__init__()
        self.n, self.vocab_size, self.mcfg, self.scfg, self.frozen = graph.n, vocab_size, mcfg, scfg, frozen
        src = torch.as_tensor(graph.src, dtype=torch.long)
        dst = torch.as_tensor(graph.dst, dtype=torch.long)
        self.register_buffer("indices", torch.stack([dst, src]))          # rows = destination
        self.register_buffer("input_nodes", torch.as_tensor(inputs, dtype=torch.long))
        self.register_buffer("output_nodes", torch.as_tensor(outputs, dtype=torch.long))

        in_deg = torch.bincount(dst, minlength=self.n).clamp(min=1).float()
        scale = 1.0 / in_deg[dst].sqrt() if mcfg.degree_normalization else torch.ones(len(dst))
        self.register_buffer("edge_scale", scale)

        # Recurrent core. Edge values come from a dedicated RNG stream so paired seeds share it (§12).
        values = torch.randn(len(src), generator=edge_generator) * mcfg.init_scale
        self.edge_values = nn.Parameter(values, requires_grad=not frozen)
        self.bias = nn.Parameter(torch.zeros(self.n), requires_grad=not frozen)
        raw_leak = torch.full((self.n,), _logit(mcfg.leak_init))
        self.raw_leak = nn.Parameter(raw_leak, requires_grad=mcfg.learned_leak and not frozen)

        # Adapters, small relative to the graph. Initialized from the global torch RNG (paired across conditions).
        self.embed = nn.Embedding(vocab_size, mcfg.embed_dim)
        self.inp = nn.Linear(mcfg.embed_dim, len(inputs))
        self.readout = nn.Linear(len(outputs), vocab_size)

    # ---- parameter groups ----------------------------------------------------------------------
    def recurrent_parameters(self):
        return [p for p in (self.edge_values, self.bias, self.raw_leak) if p.requires_grad]

    def adapter_parameters(self):
        return list(self.embed.parameters()) + list(self.inp.parameters()) + list(self.readout.parameters())

    def parameter_counts(self) -> dict:
        rec = sum(p.numel() for p in self.recurrent_parameters())
        inp = sum(p.numel() for p in list(self.embed.parameters()) + list(self.inp.parameters()))
        out = sum(p.numel() for p in self.readout.parameters())
        return {"recurrent": rec, "input": inp, "output": out, "total": rec + inp + out,
                "adapter_to_recurrent_ratio": (inp + out) / max(rec, 1)}

    # ---- dynamics ------------------------------------------------------------------------------
    @property
    def leak(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_leak)

    def effective_values(self) -> torch.Tensor:
        return self.edge_values * self.edge_scale

    def sparse_weight(self) -> torch.Tensor:
        return torch.sparse_coo_tensor(self.indices, self.effective_values().float(), (self.n, self.n), check_invariants=False)

    def init_state(self, batch: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.n, device=device or self.edge_values.device)

    def drive(self, x: torch.Tensor) -> torch.Tensor:
        """Character ids [B] -> external input [B, N] (zero except at input nodes)."""
        B = x.shape[0]
        d = torch.zeros(B, self.n, device=x.device, dtype=torch.float32)
        d[:, self.input_nodes] = self.inp(self.embed(x)).float()
        return d

    def step(self, state: torch.Tensor, x: torch.Tensor, W: torch.Tensor | None = None) -> torch.Tensor:
        W = self.sparse_weight() if W is None else W
        drive, leak = self.drive(x), self.leak
        for _ in range(self.scfg.microsteps):
            incoming = torch.sparse.mm(W, state.float().T).T          # [B, N], fp32
            proposal = torch.tanh(incoming + drive + self.bias)
            state = (1 - leak) * state + leak * proposal
        return state

    def dense_reference_step(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Slow dense path for the sparse/dense agreement test."""
        W = torch.zeros(self.n, self.n, device=state.device)
        W[self.indices[0], self.indices[1]] = self.effective_values()
        drive, leak = self.drive(x), self.leak
        for _ in range(self.scfg.microsteps):
            proposal = torch.tanh(state @ W.T + drive + self.bias)
            state = (1 - leak) * state + leak * proposal
        return state

    def logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.readout(state[:, self.output_nodes])

    def forward(self, tokens: torch.Tensor, state: torch.Tensor | None = None):
        """tokens [B, T] -> logits [B, T, V], final state [B, N]. State is zeros at sequence boundaries."""
        B, T = tokens.shape
        state = self.init_state(B, tokens.device) if state is None else state
        W = self.sparse_weight()  # build once per forward
        outs = []
        for t in range(T):
            state = self.step(state, tokens[:, t], W)
            outs.append(self.logits(state))
        return torch.stack(outs, 1), state

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int, temperature: float = 1.0, top_k: int | None = None,
                 return_states: bool = False):
        was_training = self.training
        self.eval()
        W = self.sparse_weight()
        state = self.init_state(1, prompt.device)
        states = []
        for t in range(prompt.shape[1]):
            state = self.step(state, prompt[:, t], W)
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
            state = self.step(state, nxt[:, 0], W)
            if return_states:
                states.append(state.clone())
        self.train(was_training)
        return (out, torch.cat(states, 0)) if return_states else out

    # ---- stability diagnostics (plan.md §7) -----------------------------------------------------
    @torch.no_grad()
    def stability_stats(self, state: torch.Tensor) -> dict:
        a = state.abs()
        return {"mean_abs_h": a.mean().item(), "frac_saturated": (a > 0.95).float().mean().item(),
                "act_var": state.var().item(), "mean_leak": self.leak.mean().item()}


@torch.no_grad()
def spectral_radius(model: FlyRNN, iters: int = 50, seed: int = 0) -> float:
    """Power-iteration estimate of the effective recurrent matrix's spectral radius."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(model.n, generator=g).to(model.edge_values.device)
    W = model.sparse_weight()
    lam = 0.0
    for _ in range(iters):
        v = torch.sparse.mm(W, v.unsqueeze(1)).squeeze(1)
        lam = v.norm().item()
        v = v / max(lam, 1e-12)
    return lam


def make_frozen_fly(graph: EdgeGraph, inputs, outputs, vocab_size, mcfg: ModelConfig, scfg: SequenceConfig,
                    seed: int, target_radius: float = 0.95) -> FlyRNN:
    """FrozenFly reservoir (plan.md §11): fixed random-sign, degree-normalized weights scaled to ~target_radius.
    Only the input adapter and readout train."""
    g = torch.Generator().manual_seed(seed)
    m = FlyRNN(graph, inputs, outputs, vocab_size, mcfg, scfg, edge_generator=g, frozen=True)
    with torch.no_grad():
        signs = torch.randint(0, 2, (m.edge_values.numel(),), generator=g).float() * 2 - 1
        m.edge_values.copy_(signs)
        rho = spectral_radius(m, seed=seed)
        m.edge_values.mul_(target_radius / max(rho, 1e-8))
    return m
