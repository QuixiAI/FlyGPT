#!/usr/bin/env python
"""Turn raw neuPrint tables into the two files the extractor reads.

    python data/fly/build_edges.py

Outputs:
    data/fly/neurons.parquet   body_id, type, instance, region   (region: central_brain | vnc | other)
    data/fly/edges.parquet     src, dst, weight                   (body ids, synapse count)

The `region` mapping is the one place a wrong choice turns "fly brain" into "fly spinal cord"
(README §3.1). Set --region-column to the v1.0 annotation column that separates brain from VNC
and check the printed breakdown before running build_graph.py.
"""
from __future__ import annotations

import argparse

import pandas as pd

BRAIN_TOKENS = ("brain", "central", "cb", "cns_brain")
VNC_TOKENS = ("vnc", "ventral", "neuromere", "thoracic", "abdominal")


def coarse_region(value) -> str:
    v = str(value).lower()
    if any(t in v for t in VNC_TOKENS):
        return "vnc"
    if any(t in v for t in BRAIN_TOKENS):
        return "central_brain"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-connections", default="data/fly/raw_connections.parquet")
    ap.add_argument("--raw-neurons", default="data/fly/raw_neurons.parquet")
    ap.add_argument("--region-column", default=None, help="annotation column to derive region from")
    ap.add_argument("--out-dir", default="data/fly")
    args = ap.parse_args()

    conns = pd.read_parquet(args.raw_connections)
    conns = conns[conns["bodyId_pre"] != conns["bodyId_post"]]
    edges = conns.rename(columns={"bodyId_pre": "src", "bodyId_post": "dst"})[["src", "dst", "weight"]]
    edges.to_parquet(f"{args.out_dir}/edges.parquet", index=False)

    neurons = pd.read_parquet(args.raw_neurons)
    keep = [c for c in ["bodyId", "type", "instance"] if c in neurons.columns]
    out = neurons[keep].rename(columns={"bodyId": "body_id"})
    if args.region_column and args.region_column in neurons.columns:
        out["region"] = neurons[args.region_column].map(coarse_region)
    else:
        print(f"no region column given/found (columns: {list(neurons.columns)}); region left unset")
    out.to_parquet(f"{args.out_dir}/neurons.parquet", index=False)

    n_nodes = pd.concat([edges["src"], edges["dst"]]).nunique()
    print(f"{n_nodes:,} neurons with connections, {len(edges):,} directed edges")
    if "region" in out.columns:
        print(out["region"].value_counts().to_string())


if __name__ == "__main__":
    main()
