#!/usr/bin/env python
"""Train one condition for one seed.

    python train.py configs/launch.yaml --condition real --seed 1
    python train.py configs/launch.yaml --condition degree_preserving --seed 1
    python train.py configs/launch.yaml --condition frozen --seed 1          # reservoir gate
    python train.py configs/launch.yaml --condition rnn|gru|transformer --seed 1
    torchrun --nproc_per_node 8 train.py configs/full_cns.yaml --condition real --seed 1     # data-parallel: 8 GPUs, one model

Data parallel (launched with torchrun): every rank holds a full copy of the model, draws its own batch of
`batch_size` windows from a rank-specific data-order stream (seed*1000 + rank, so paired conditions still share
it), and gradients are all-reduced, so one optimizer step sees world_size × batch_size windows. Logging,
evaluation, generations and checkpoints happen on rank 0 only.

Writes runs/<project>/<graph_name>/<condition>_seed<k>/{log.jsonl, generations.jsonl, meta.json}
and checkpoints/<project>/<graph_name>/<condition>_seed<k>.pt (best val).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flygpt import Config, build_model
from flygpt.checkpoint import save_checkpoint
from flygpt.data import make_streams


def ddp_setup():
    """Returns (rank, world_size, local_rank); initializes the process group when launched by torchrun."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        return dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    return 0, 1, 0


def pick_device(name, local_rank=0):
    if name:
        return torch.device(name)
    if torch.cuda.is_available() and "WORLD_SIZE" in os.environ:
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, stream, device, n_batches):
    model.eval()
    tot = 0.0
    for _ in range(n_batches):
        x, y = stream.batch(device)
        lg, _ = model(x)
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1)).item()
    model.train()
    return tot / n_batches


@torch.no_grad()
def fixed_prompt_generations(model, vocab, device, cfg: Config, seed: int):
    out = {}
    for p in cfg.evaluation.prompts:
        if any(c not in vocab.chars for c in p):
            out[p] = None  # prompt uses a character outside this corpus's vocabulary
            continue
        torch.manual_seed(seed)
        ids = torch.tensor([vocab.encode(p)], device=device)
        gen = model.generate(ids, cfg.evaluation.generation_chars, cfg.evaluation.generation_temperature)
        out[p] = vocab.decode(gen[0].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/launch.yaml")
    ap.add_argument("--condition", default="real")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.steps:
        cfg.training.steps = args.steps
    rank, world, local_rank = ddp_setup()
    device = pick_device(args.device, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    T = cfg.training
    main_rank = rank == 0

    vocab, train_stream, val_stream = make_streams(cfg.dataset, cfg.sequence.context, T.batch_size,
                                                   args.seed if world == 1 else args.seed * 1000 + rank)
    model, meta = build_model(cfg, args.condition, args.seed, len(vocab))
    model.to(device)
    is_fly = hasattr(model, "stability_stats")
    raw = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    run_name = f"{args.condition}_seed{args.seed}"
    run_dir = Path("runs") / cfg.project / cfg.graph_name / run_name
    ckpt_path = Path("checkpoints") / cfg.project / cfg.graph_name / f"{run_name}.pt"
    if main_rank:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "meta.json").write_text(json.dumps({"config": cfg.to_dict(), "seed": args.seed, "device": str(device),
                                                       "world_size": world, "global_batch": world * T.batch_size, **meta}, indent=2, default=str))
        print(f"{run_name} device={device} world={world} global_batch={world * T.batch_size} vocab={len(vocab)} params={meta.get('params', meta)}")

    groups = [{"params": raw.adapter_parameters(), "lr": T.adapter_lr}]
    if raw.recurrent_parameters():
        groups.append({"params": raw.recurrent_parameters(), "lr": T.recurrent_lr})
    opt = torch.optim.AdamW(groups, weight_decay=T.weight_decay)

    gen_steps = {int(round(f * T.steps)) for f in cfg.evaluation.generation_fractions} - {0}
    log = (run_dir / "log.jsonl").open("w") if main_rank else None
    gens = (run_dir / "generations.jsonl").open("w") if main_rank else None
    best = float("inf")
    chars_seen, t0 = 0, time.time()

    def record_generations(step):
        if not main_rank:
            return
        g = fixed_prompt_generations(raw, vocab, device, cfg, args.seed)
        gens.write(json.dumps({"step": step, "generations": g}) + "\n"); gens.flush()
        first = next((v for v in g.values() if v), "")
        print(f"--- generations @ step {step} ---\n    " + first[:200].replace("\n", "\n    ") + "\n")

    record_generations(0)
    model.train()
    for step in range(1, T.steps + 1):
        x, y = train_stream.batch(device)
        lg, state = model(x)
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), T.grad_clip).item()
        opt.step()
        chars_seen += x.numel() * world

        if world > 1:  # train loss averaged over ranks, so the log reflects the global batch
            loss = loss.detach()
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        rec = {"step": step, "train_loss": loss.item(), "chars_seen": chars_seen, "wall": time.time() - t0,
               "tokens_per_sec": chars_seen / max(time.time() - t0, 1e-6), "grad_norm": gnorm}
        if device.type == "cuda":
            rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if is_fly and state is not None:
            rec.update(raw.stability_stats(state))
        if step % T.eval_every == 0 or step == T.steps:
            rec["val_loss"] = evaluate(raw, val_stream, device, T.eval_batches) if main_rank else 0.0
            if main_rank and rec["val_loss"] < best:
                best = rec["val_loss"]
                save_checkpoint(ckpt_path, raw, cfg, vocab, args.condition, args.seed, step, best, meta)
            if main_rank:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}))
        if main_rank:
            log.write(json.dumps(rec) + "\n"); log.flush()
        if step in gen_steps:
            record_generations(step)

    if main_rank:
        print(f"done. best val {best:.4f} -> {ckpt_path}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
