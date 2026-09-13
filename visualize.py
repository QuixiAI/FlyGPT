#!/usr/bin/env python
"""Static activity plot: sampled neuron states over a prompt + generation.

python visualize.py --ckpt checkpoints/fly_5k.pt --prompt "ROMEO:" --out runs/activity.png
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from flygpt.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--n-neurons", type=int, default=200, help="how many neurons to plot")
    ap.add_argument("--out", default="runs/activity.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.manual_seed(0)
    model, graph, vocab, _, _ = load_checkpoint(args.ckpt)
    ids = torch.tensor([vocab.encode(args.prompt)])
    out, states = model.generate(ids, args.max_new, temperature=0.8, return_states=True)
    text = vocab.decode(out[0].tolist())

    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(graph.n, min(args.n_neurons, graph.n), replace=False))
    act = states[:, pick].numpy().T  # [neurons, time]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(act, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xlabel("character step")
    ax.set_ylabel("sampled fly neuron")
    ax.set_title(f"FlyGPT activity  |  {text[:60]!r}...")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}\n\n{text}")


if __name__ == "__main__":
    main()
