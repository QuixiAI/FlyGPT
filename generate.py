#!/usr/bin/env python
"""python generate.py --ckpt checkpoints/flygpt-v0/cb5k/real_seed1.pt --prompt "ROMEO:" """
import argparse
import torch
from flygpt.checkpoint import load_checkpoint

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
model, _, vocab, _, _ = load_checkpoint(args.ckpt, args.device)
ids = torch.tensor([vocab.encode(args.prompt)], device=args.device)
print(vocab.decode(model.generate(ids, args.max_new, args.temperature, args.top_k)[0].tolist()))
