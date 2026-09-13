#!/usr/bin/env python
"""Evaluate checkpoints and append rows to results/scoreboard.md.

python evaluate.py checkpoints/fly_5k.pt checkpoints/scrambled_5k.pt ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from flygpt.checkpoint import load_checkpoint
from flygpt.data import make_streams
from flygpt.model import count_params
from train import evaluate, sample

SCOREBOARD = Path("results/scoreboard.md")
HEADER = "| Model | Neurons | Edges | Params | Val loss | Sample |\n|---|--:|--:|--:|--:|---|\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--prompt", default="ROMEO:")
    args = ap.parse_args()

    if not SCOREBOARD.exists() or HEADER.split("\n")[0] not in SCOREBOARD.read_text():
        SCOREBOARD.write_text("# Scoreboard\n\n" + HEADER)

    torch.manual_seed(0)
    for path in args.ckpts:
        model, graph, vocab, cfg, _ = load_checkpoint(path, args.device)
        _, _, val = make_streams(cfg.data, cfg.train.seed)
        loss = evaluate(model, val, torch.device(args.device), args.batches)
        text = sample(model, vocab, torch.device(args.device), args.prompt, 80)
        text = text[len(args.prompt):].replace("\n", " / ").replace("|", "\\|").strip()
        n = graph.n if graph else "-"
        e = graph.n_edges if graph else "-"
        row = f"| {cfg.name} | {n} | {e} | {count_params(model):,} | {loss:.3f} | `{text}` |\n"
        SCOREBOARD.open("a").write(row)
        print(row, end="")


if __name__ == "__main__":
    main()
