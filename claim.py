#!/usr/bin/env python
"""Apply the pre-registered paired-seed claim rule (README §12).

    python claim.py configs/launch.yaml [--control degree_preserving] [--seeds 1 2 3 4 5]
"""
import argparse, json
from pathlib import Path
from flygpt import Config
from flygpt.analysis import paired_differences, apply_claim_rule

ap = argparse.ArgumentParser()
ap.add_argument("config", nargs="?", default="configs/launch.yaml")
ap.add_argument("--control", default="degree_preserving")
ap.add_argument("--seeds", type=int, nargs="*")
ap.add_argument("--metric", default="final_val")
args = ap.parse_args()
cfg = Config.load(args.config)
seeds = args.seeds or cfg.evaluation.seeds_claim
root = Path("runs") / cfg.project / cfg.graph_name
paired = paired_differences(root, "real", args.control, seeds, args.metric)
verdict = apply_claim_rule(paired, cfg.evaluation.claim_rule, len(cfg.evaluation.seeds_claim))
for r in paired["rows"]:
    print(f"seed {r['seed']}: real {r['real']:.4f}  {args.control} {r['control']:.4f}  Δ {r['delta']:+.4f}")
print(f"mean Δ {paired['mean_delta']:+.4f}  ->  {verdict['verdict']}")
Path("results").mkdir(exist_ok=True)
Path(f"results/claim_{args.control}.json").write_text(json.dumps({"paired": paired, "verdict": verdict}, indent=2))
