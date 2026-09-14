"""Fused CUDA recurrence for FlyRNN (plan.md §7, §8): flygpt/csrc/fly_recurrence.cu bound through
torch.utils.cpp_extension. One character window (T chars × microsteps) is a single autograd Function;
the C++ host loop launches one fused kernel per microstep forward and three backward, so a training
step is no longer CPU-launch-bound. The math is identical to FlyRNN.step; tests/test_kernels.py checks
logits and every gradient against the PyTorch sparse path.

Build: JIT on first use (cached under ~/.cache/torch_extensions). Requires nvcc (CUDA_HOME) matching
torch's CUDA major version.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        if not os.environ.get("CUDA_HOME") and Path("/usr/local/cuda").exists():
            os.environ["CUDA_HOME"] = "/usr/local/cuda"
        cc = torch.cuda.get_device_capability(0)
        _ext = load(name="fly_recurrence", sources=[str(Path(__file__).parent / "csrc" / "fly_recurrence.cu")],
                    extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{cc[0]}{cc[1]},code=sm_{cc[0]}{cc[1]}"],
                    verbose=False)
    return _ext


def available() -> bool:
    try:
        return torch.cuda.is_available() and _load() is not None
    except Exception:
        return False


class CSRGraph:
    """Fixed sparse structure on the device: CSR by destination (rows = dst), its transpose, and the
    input-neuron position map. Edge order inside is CSR order; `perm` maps model edge order -> CSR order."""

    def __init__(self, src: torch.Tensor, dst: torch.Tensor, n: int, input_nodes: torch.Tensor):
        dev = src.device
        self.n, self.E = n, int(src.numel())
        self.perm = torch.argsort(dst * n + src)
        self.src = src[self.perm].to(torch.int32).contiguous()
        self.dst = dst[self.perm].to(torch.int32).contiguous()
        crow = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        crow[1:] = torch.cumsum(torch.bincount(self.dst.long(), minlength=n), 0)
        self.crow = crow.to(torch.int32).contiguous()
        permT = torch.argsort(self.src.long() * n + self.dst.long())
        self.permT = permT.to(torch.int32).contiguous()
        self.colT = self.dst[permT].contiguous()
        crowT = torch.zeros(n + 1, dtype=torch.int64, device=dev)
        crowT[1:] = torch.cumsum(torch.bincount(self.src.long(), minlength=n), 0)
        self.crowT = crowT.to(torch.int32).contiguous()
        in_pos = torch.full((n,), -1, dtype=torch.int64, device=dev)
        in_pos[input_nodes] = torch.arange(input_nodes.numel(), device=dev)
        self.in_pos = in_pos.to(torch.int32).contiguous()
        self.n_in = int(input_nodes.numel())


class FlyRecurrence(torch.autograd.Function):
    """out[T, B, N] = state after each of T characters, running M microsteps per character from state0 [B, N]
    with drives [T, n_in, B] (external input to the input neurons)."""

    @staticmethod
    def forward(ctx, vals_model_order, leak, bias, drives, state0, graph: CSRGraph, microsteps: int):
        ext = _load()
        vals = vals_model_order[graph.perm].contiguous().float()
        leak_c, bias_c, drives_c = leak.contiguous().float(), bias.contiguous().float(), drives.contiguous().float()
        states, props = ext.forward(vals, graph.src, graph.crow, graph.in_pos, leak_c, bias_c, drives_c,
                                    state0.T.contiguous().float(), microsteps)
        ctx.save_for_backward(vals, leak_c, states, props)
        ctx.graph, ctx.microsteps = graph, microsteps
        return states[microsteps::microsteps].transpose(1, 2)

    @staticmethod
    def backward(ctx, g_out):
        vals, leak, states, props = ctx.saved_tensors
        g, M = ctx.graph, ctx.microsteps
        g_vals, g_leak, g_bias, g_drives, g_s0 = _load().backward(
            vals, g.src, g.dst, g.permT, g.colT, g.crowT, g.in_pos, leak, states, props,
            g_out.transpose(1, 2).contiguous().float(), M, g.n_in)
        g_vals_model = torch.empty_like(g_vals)
        g_vals_model[g.perm] = g_vals
        return g_vals_model, g_leak, g_bias, g_drives, g_s0.T, None, None
