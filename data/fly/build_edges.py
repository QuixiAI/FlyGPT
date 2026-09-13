#!/usr/bin/env python
"""Turn the raw neuPrint tables into the two files the model reads.

    python data/fly/build_edges.py

Outputs:
    data/fly/neurons.parquet   columns: body_id, type, instance  (whatever was available)
    data/fly/edges.parquet     columns: src, dst, weight          (body ids, synapse count)

The graph loader (flygpt/graph.py) only needs edges.parquet; neurons.parquet is
for later region/type-based experiments and for labelling the visualization.
"""
from __future__ import annotations

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-connections", default="data/fly/raw_connections.parquet")
    ap.add_argument("--raw-neurons", default="data/fly/raw_neurons.parquet")
    ap.add_argument("--min-synapses", type=int, default=1)
    ap.add_argument("--out-dir", default="data/fly")
    args = ap.parse_args()

    conns = pd.read_parquet(args.raw_connections)
    conns = conns[conns["weight"] >= args.min_synapses]
    conns = conns[conns["bodyId_pre"] != conns["bodyId_post"]]
    edges = conns.rename(columns={"bodyId_pre": "src", "bodyId_post": "dst"})[["src", "dst", "weight"]]
    edges.to_parquet(f"{args.out_dir}/edges.parquet", index=False)

    neurons = pd.read_parquet(args.raw_neurons)
    keep = [c for c in ["bodyId", "type", "instance", "somaLocation"] if c in neurons.columns]
    neurons[keep].rename(columns={"bodyId": "body_id"}).to_parquet(f"{args.out_dir}/neurons.parquet", index=False)

    n_nodes = pd.concat([edges["src"], edges["dst"]]).nunique()
    print(f"{n_nodes:,} neurons with connections, {len(edges):,} directed edges")


if __name__ == "__main__":
    main()
