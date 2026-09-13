#!/usr/bin/env python
"""Build-order steps 2-5: extract the dense core, build every control for every claim seed, select I/O
by degree, run diagnostics and the path-length gate on each condition.

    python build_graph.py configs/launch.yaml [--synthetic]   # --synthetic: fake connectome, for smoke tests

Writes graphs/<graph_name>/:
    subgraph_node_ids.txt  subgraph_edges.pt  subgraph_config.yaml  subgraph_hash.txt  subgraph_stats.json
    real/edges.pt  real/diagnostics.json
    degree_preserving_seed<k>/edges.pt + diagnostics.json  (one per claim seed)
    uniform_random_seed<k>/...                              (if listed in controls)
    gate.json                                               # pass/fail per condition
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flygpt import Config
from flygpt.connectome import from_edge_table, extract_dense_core, save_subgraph, build_control, select_io, diagnose, path_gate
from flygpt.connectome.extract import graph_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/launch.yaml")
    ap.add_argument("--synthetic", action="store_true", help="use a synthetic connectome instead of MaleCNS")
    ap.add_argument("--seeds", type=int, nargs="*", help="override claim seeds")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    gcfg = cfg.graph

    if args.synthetic:
        from flygpt.connectome.synthetic import synthetic_edge_table
        src_ids, dst_ids, w, regions = synthetic_edge_table(n=max(4 * gcfg.target_neurons, 400), density=0.02, seed=0)
        cand = np.array([i for i, r in regions.items() if r == gcfg.region_filter or gcfg.region_filter in ("whole_cns", "none")])
        source = "synthetic"
    else:
        from flygpt.connectome.malecns import load_candidate_pool
        src_ids, dst_ids, w, cand, regions = load_candidate_pool(gcfg)
        source = gcfg.source

    pool = from_edge_table(src_ids, dst_ids, w, candidate_ids=cand)
    print(f"candidate pool ({gcfg.region_filter}): {pool.n:,} neurons, {pool.n_edges:,} edges")
    core, log = extract_dense_core(pool, gcfg.target_neurons, gcfg.min_synapses, gcfg.retain_largest_scc)
    log["source"] = source
    out = cfg.graph_dir
    stats = save_subgraph(core, gcfg, out, extra_stats=log)
    stats["region_breakdown"] = graph_stats(core, regions).get("region_breakdown")
    (out / "subgraph_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"subgraph: {core.n:,} neurons, {core.n_edges:,} edges, core_k={log['core_k']}, "
          f"scc={stats['largest_scc_fraction']:.3f}, regions={stats['region_breakdown']}")

    seeds = args.seeds or cfg.evaluation.seeds_claim
    gate_results = {}
    conditions = [("real", 0)] + [(c, k) for c in cfg.controls if c != "real" for k in seeds]
    for name, seed in conditions:
        g, prov = build_control(core, name, seed)
        d = out / ("real" if name == "real" else f"{name}_seed{seed}")
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"src": torch.as_tensor(g.src), "dst": torch.as_tensor(g.dst), "weight": torch.as_tensor(g.weight),
                    "ids": torch.as_tensor(g.ids), "n": g.n}, d / "edges.pt")
        inp, outp, io = select_io(g, cfg.interface.input_nodes, cfg.interface.input_rule,
                                  cfg.interface.output_nodes, cfg.interface.output_rule)
        np.savetxt(d / "input_nodes.txt", inp, fmt="%d")
        np.savetxt(d / "output_nodes.txt", outp, fmt="%d")
        diag = diagnose(g, inp, outp)
        gate = path_gate(diag, cfg.sequence.microsteps, cfg.gate)
        (d / "diagnostics.json").write_text(json.dumps({"provenance": prov, "io": io, "diagnostics": diag, "gate": gate}, indent=2))
        gate_results[d.name] = gate["passed"]
        extra = f" survival={prov['edge_survival']:.3f} plateau_agrees={prov['plateau_agrees']}" if name == "degree_preserving" else ""
        print(f"{d.name:28s} scc={diag['largest_scc_fraction']:.3f} recip={diag['reciprocal_pairs']:6d} "
              f"reach={diag['io_reachable_fraction']:.3f} p90={diag['io_path_p90']} gate={'PASS' if gate['passed'] else 'FAIL'}{extra}")

    # I/O sets must be byte-identical across real and degree_preserving (§5)
    real_in = np.loadtxt(out / "real/input_nodes.txt", dtype=int)
    for k in seeds:
        dp = out / f"degree_preserving_seed{k}/input_nodes.txt"
        if dp.exists():
            assert np.array_equal(real_in, np.loadtxt(dp, dtype=int)), "I/O sets differ between real and degree_preserving"
    (out / "gate.json").write_text(json.dumps(gate_results, indent=2))
    print("all gates passed" if all(gate_results.values()) else "GATE FAILED: fix the subgraph or interface, do not tune microsteps")


if __name__ == "__main__":
    main()
