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
ap.add_argument("--metric", default="final_val", choices=["final_val", "best_val"], help="metric for the primary column")
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
        r = read_run(d)
        by_cond[cond].append((int(seed), r["final_val"], r["best_val"], r["steps"]))
        p = json.loads((d / "meta.json").read_text()).get("params", {})
        params[cond] = p.get("total", "-") if isinstance(p, dict) else "-"
    except (ValueError, FileNotFoundError):
        pass

lines = ["# Scoreboard", "", f"Config: `{args.config}` · graph `{cfg.graph_name}` · metric `{args.metric}` (nats/char). ",
         "Same data, same split, same context, same training loop for every row.", ""]
if ref:
    lines += [f"Measured on our split: unigram {ref['unigram_nats']:.3f}, bigram {ref['bigram_nats']:.3f}.",
              "External reference: nanoGPT `train_shakespeare_char` best val ~1.47 (verify + cite at launch).", ""]
lines += ["| Model | Params | Final val (mean) | Range | Best val (mean) | Range | Seeds | Notes |", "|---|--:|--:|---|--:|---|--:|---|"]
NOTES = {"real": "real fly wiring, edges trained", "degree_preserving": "same neurons/degrees, wiring scrambled",
         "frozen": "real wiring, edges frozen (reservoir); adapters only", "rnn": "dense tanh RNN, parameter-matched",
         "gru": "GRU, parameter-matched", "transformer": "tiny Transformer, parameter-matched"}
for cond, rows in by_cond.items():
    fin = [v for _, v, _, _ in rows]; best = [v for _, _, v, _ in rows]
    note = NOTES.get(cond, "")
    if max(fin) - min(best) > 0.1:
        note += "; overfits after its best step"
    ps = f"{params[cond]:,}" if isinstance(params[cond], int) else str(params[cond])
    lines.append(f"| {cond} | {ps} | {sum(fin)/len(fin):.3f} | {min(fin):.3f}–{max(fin):.3f} | {sum(best)/len(best):.3f} | {min(best):.3f}–{max(best):.3f} | {len(rows)} | {note} |")
Path("results/scoreboard.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
