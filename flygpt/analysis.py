"""Paired-seed decision rule (plan.md §12).

    Δ_k = loss(Scrambled_k) − loss(Real_k)

"Wiring matters" is claimed only if every Δ_k has the same sign AND mean Δ >= min_mean_diff_nats.
Otherwise the result is "no detectable difference at this scale". The rule is fixed in the launch
config before any result is seen.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import ClaimRule


def read_run(run_dir: str | Path) -> dict:
    """Summarize a run directory: final and best val loss from log.jsonl."""
    rows = [json.loads(l) for l in (Path(run_dir) / "log.jsonl").read_text().splitlines() if l.strip()]
    vals = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    if not vals:
        raise ValueError(f"no val_loss rows in {run_dir}")
    return {"steps": vals[-1][0], "final_val": vals[-1][1], "best_val": min(v for _, v in vals),
            "trace": vals}


def paired_differences(runs_root: str | Path, real: str, control: str, seeds: list[int], metric: str = "final_val") -> dict:
    root = Path(runs_root)
    rows = []
    for k in seeds:
        r = read_run(root / f"{real}_seed{k}")[metric]
        c = read_run(root / f"{control}_seed{k}")[metric]
        rows.append({"seed": k, "real": r, "control": c, "delta": c - r})
    deltas = [x["delta"] for x in rows]
    return {"real": real, "control": control, "metric": metric, "rows": rows,
            "mean_real": sum(x["real"] for x in rows) / len(rows),
            "mean_control": sum(x["control"] for x in rows) / len(rows),
            "mean_delta": sum(deltas) / len(deltas), "min_delta": min(deltas), "max_delta": max(deltas)}


def apply_claim_rule(paired: dict, rule: ClaimRule, n_required: int) -> dict:
    deltas = [x["delta"] for x in paired["rows"]]
    same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
    enough = len(deltas) >= n_required
    big_enough = abs(paired["mean_delta"]) >= rule.min_mean_diff_nats
    passed = enough and (same_sign or not rule.all_paired_diffs_same_sign) and big_enough
    direction = "real_better" if paired["mean_delta"] > 0 else "control_better"
    return {"n_seeds": len(deltas), "n_required": n_required, "all_same_sign": same_sign,
            "mean_delta": paired["mean_delta"], "min_mean_diff_nats": rule.min_mean_diff_nats,
            "passed": passed, "direction": direction if passed else None,
            "verdict": (f"wiring matters ({direction})" if passed else "no detectable difference at this scale")}
