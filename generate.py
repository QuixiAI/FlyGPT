#!/usr/bin/env python
"""python generate.py --ckpt checkpoints/fly_5k.pt --prompt "ROMEO:" """
from __future__ import annotations

import argparse

import torch

from flygpt.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, _, vocab, _, ckpt = load_checkpoint(args.ckpt, args.device)
    ids = torch.tensor([vocab.encode(args.prompt)], device=args.device)
    out = model.generate(ids, args.max_new, args.temperature, args.top_k)
    print(vocab.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
