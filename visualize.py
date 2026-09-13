#!/usr/bin/env python
"""Static neural-activity raster for a prompt (plan.md §17; sampled neurons, explanatory only).

    python visualize.py --ckpt checkpoints/flygpt-v0/cb5k/real_seed1.pt --prompt "ROMEO:" --out results/activity.png
"""
import argparse
import numpy as np, torch
from flygpt.checkpoint import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--prompt", default="ROMEO:")
ap.add_argument("--max-new", type=int, default=100)
ap.add_argument("--n-neurons", type=int, default=200)
ap.add_argument("--out", default="results/activity.png")
args = ap.parse_args()
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
model, _, vocab, _, _ = load_checkpoint(args.ckpt)
ids = torch.tensor([vocab.encode(args.prompt)])
out, states = model.generate(ids, args.max_new, temperature=0.8, return_states=True)
text = vocab.decode(out[0].tolist())
pick = np.sort(np.random.default_rng(0).choice(model.n, min(args.n_neurons, model.n), replace=False))
fig, ax = plt.subplots(figsize=(14, 6))
ax.imshow(states[:, pick].numpy().T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("character step"); ax.set_ylabel("sampled fly neuron")
ax.set_title(f"FlyGPT activity (sampled, not a biological simulation)  |  {text[:60]!r}...")
fig.tight_layout(); fig.savefig(args.out, dpi=120)
print(f"wrote {args.out}\n\n{text}")
