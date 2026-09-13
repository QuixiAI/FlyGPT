#!/usr/bin/env python
"""Train any config: python train.py configs/fly_5k.yaml [--steps N] [--device cpu|cuda|mps]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from flygpt import Config, build_model
from flygpt.data import make_streams
from flygpt.model import count_params


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, stream, device, n_batches: int = 10) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = stream.batch(device)
        logits, _ = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample(model, vocab, device, prompt: str = "ROMEO:", max_new: int = 200) -> str:
    ids = torch.tensor([vocab.encode(prompt)], device=device)
    out = model.generate(ids, max_new=max_new, temperature=0.8)
    model.train()
    return vocab.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--device")
    ap.add_argument("--prompt", default="ROMEO:")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.steps:
        cfg.train.steps = args.steps
    torch.manual_seed(cfg.train.seed)
    device = pick_device(args.device)

    vocab, train_stream, val_stream = make_streams(cfg.data, cfg.train.seed)
    model, graph = build_model(cfg, len(vocab))
    model.to(device)

    run_dir = Path("runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print(f"run={cfg.name} model={cfg.model_type} device={device} vocab={len(vocab)} params={count_params(model):,}")
    if graph is not None:
        print(f"graph: n={graph.n} edges={graph.n_edges} in={len(graph.input_nodes)} out={len(graph.output_nodes)} scrambled={cfg.graph.scramble}")

    opt = torch.optim.AdamW([
        {"params": model.recurrent_parameters(), "lr": cfg.train.lr_recurrent},
        {"params": model.adapter_parameters(), "lr": cfg.train.lr_adapters},
    ], weight_decay=cfg.train.weight_decay)

    use_bf16 = cfg.train.bf16 and device.type == "cuda"
    log = open(run_dir / "log.jsonl", "a")
    best_val = float("inf")
    t0 = time.time()
    model.train()
    for step in range(1, cfg.train.steps + 1):
        x, y = train_stream.batch(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

        rec = {"step": step, "train_loss": loss.item(), "t": time.time() - t0}
        if step % cfg.train.eval_every == 0 or step == cfg.train.steps:
            rec["val_loss"] = evaluate(model, val_stream, device)
            if rec["val_loss"] < best_val:
                best_val = rec["val_loss"]
                torch.save({"config": cfg.to_dict(), "vocab": vocab.chars, "model": model.state_dict(),
                            "step": step, "val_loss": best_val}, ckpt_dir / f"{cfg.name}.pt")
        if step % cfg.train.sample_every == 0 or step == cfg.train.steps:
            text = sample(model, vocab, device, args.prompt)
            (run_dir / "samples.txt").open("a").write(f"\n=== step {step} ===\n{text}\n")
            print(f"--- sample @ {step} ---\n{text}\n")
        if "val_loss" in rec or step % 50 == 0:
            print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        log.flush()

    print(f"done. best val loss {best_val:.4f} -> checkpoints/{cfg.name}.pt")


if __name__ == "__main__":
    main()
