#!/usr/bin/env python
"""Scoreboard (plan.md §11): model, params, val loss mean ± range over seeds, notes.

    python evaluate.py configs/launch.yaml            # reads runs/, writes results/scoreboard.md
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from flygpt import Config
from flygpt.analysis import read_run

ap = argparse.ArgumentParser()
ap.add_argument("config", nargs="?", default="configs/launch.yaml")
ap.add_argument("--metric", default="final_val", choices=["final_val", "best_val"])
args = ap.parse_args()
cfg = Config.load(args.config)
root = Path("runs") / cfg.project / cfg.graph_name

split = Path("data/shakespeare/split.json")
ref = json.loads(split.read_text()) if split.exists() else {}
by_cond = defaultdict(list)
params = {}
for d in sorted(root.glob("*_seed*")):
    cond, seed = d.name.rsplit("_seed", 1)
    try:
        by_cond[cond].append((int(seed), read_run(d)[args.metric]))
        params[cond] = json.loads((d / "meta.json").read_text()).get("params", {}).get("total", "-")
    except (ValueError, FileNotFoundError):
        pass

lines = ["# Scoreboard", "", f"Config: `{args.config}` · graph `{cfg.graph_name}` · metric `{args.metric}` (nats/char). ",
         "Same data, same split, same context, same training loop for every row.", ""]
if ref:
    lines += [f"Measured on our split: unigram {ref['unigram_nats']:.3f}, bigram {ref['bigram_nats']:.3f}.",
              "External reference: nanoGPT `train_shakespeare_char` best val ~1.47 (verify + cite at launch).", ""]
lines += ["| Model | Params | Val loss (mean) | Range | Seeds | Notes |", "|---|--:|--:|---|--:|---|"]
for cond, rows in by_cond.items():
    vals = [v for _, v in rows]
    lines.append(f"| {cond} | {params[cond]:,} | {sum(vals)/len(vals):.3f} | {min(vals):.3f}–{max(vals):.3f} | {len(vals)} | |"
                 if isinstance(params[cond], int) else
                 f"| {cond} | {params[cond]} | {sum(vals)/len(vals):.3f} | {min(vals):.3f}–{max(vals):.3f} | {len(vals)} | |")
Path("results/scoreboard.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
