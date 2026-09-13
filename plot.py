#!/usr/bin/env python
"""Launch chart, regenerated from logs (plan.md §9): val loss vs step, wall-clock, and characters seen.

    python plot.py configs/launch.yaml --out results/val_loss.png
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
from flygpt import Config

ap = argparse.ArgumentParser()
ap.add_argument("config", nargs="?", default="configs/launch.yaml")
ap.add_argument("--out", default="results/val_loss.png")
args = ap.parse_args()
cfg = Config.load(args.config)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("runs") / cfg.project / cfg.graph_name
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
colors = {}
for d in sorted(root.glob("*_seed*")):
    cond = d.name.rsplit("_seed", 1)[0]
    rows = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines() if '"val_loss"' in l]
    if not rows:
        continue
    c = colors.setdefault(cond, f"C{len(colors)}")
    for ax, key in zip(axes, ["step", "wall", "chars_seen"]):
        ax.plot([r[key] for r in rows], [r["val_loss"] for r in rows], color=c, alpha=0.7,
                label=cond if d.name.endswith("seed1") or cond not in [l.get_label() for l in ax.lines] else None)
split = Path("data/shakespeare/split.json")
if split.exists():
    ref = json.loads(split.read_text())
    for ax in axes:
        ax.axhline(ref["bigram_nats"], ls="--", color="gray", lw=1, label="bigram")
for ax, lab in zip(axes, ["step", "wall-clock (s)", "characters seen"]):
    ax.set_xlabel(lab); ax.grid(alpha=0.3)
axes[0].set_ylabel("val loss (nats/char)"); axes[0].legend()
fig.suptitle(f"{cfg.project} / {cfg.graph_name}")
fig.tight_layout(); Path(args.out).parent.mkdir(exist_ok=True); fig.savefig(args.out, dpi=130)
print("wrote", args.out)
