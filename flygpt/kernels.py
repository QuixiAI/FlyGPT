"""Fused CUDA recurrence for FlyRNN (plan.md §7, §8), provided by the connectome-kernels package
(https://github.com/QuixiAI/connectome-kernels). One character window (T chars × microsteps) is a single
autograd Function; the C++ host loop launches one fused kernel per microstep forward and three backward, so
a training step is no longer CPU-launch-bound. The math is identical to FlyRNN.step; tests/test_kernels.py
checks logits and every gradient against the PyTorch sparse path.
"""
from __future__ import annotations

import torch

def _load():
    """The compiled extension from the connectome-kernels package (pip install connectome-kernels)."""
    from connectome_kernels import recurrence
    return recurrence._load()


def available() -> bool:
    try:
        from connectome_kernels import available as _avail
        return torch.cuda.is_available() and _avail()
    except Exception:
        return False


from connectome_kernels import SparseGraph as CSRGraph, SparseRecurrence as FlyRecurrence  # noqa: E402

__all__ = ["CSRGraph", "FlyRecurrence", "available"]
