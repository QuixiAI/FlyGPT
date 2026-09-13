#!/usr/bin/env python
"""Fetch the MaleCNS connectivity table from neuPrint.

Requires a neuPrint token (https://neuprint.janelia.org, Account -> Auth Token):
    export NEUPRINT_TOKEN=...
    python data/fly/fetch_malecns.py --out data/fly/raw_connections.parquet

Writes one row per (bodyId_pre, bodyId_post) with the synapse count, plus a
neurons table with type/instance/region annotations when available.

TODO: confirm the exact dataset name on neuPrint ("male-cns:v0.9" or similar)
and whether the full table is small enough to pull via Cypher in pages, or
whether the bulk download from the MaleCNS release page is the better path.
"""
from __future__ import annotations

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="male-cns")
    ap.add_argument("--server", default="https://neuprint.janelia.org")
    ap.add_argument("--min-synapses", type=int, default=1)
    ap.add_argument("--out", default="data/fly/raw_connections.parquet")
    ap.add_argument("--neurons-out", default="data/fly/raw_neurons.parquet")
    args = ap.parse_args()

    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit("set NEUPRINT_TOKEN (neuprint.janelia.org -> Account -> Auth Token)")

    from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria  # pip install neuprint-python

    c = Client(args.server, dataset=args.dataset, token=token)
    print(c.fetch_version())
    neurons, conns = fetch_adjacencies(NeuronCriteria(), NeuronCriteria(), min_total_weight=args.min_synapses)
    conns = conns.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    conns.to_parquet(args.out, index=False)
    neurons.to_parquet(args.neurons_out, index=False)
    print(f"{len(neurons):,} neurons, {len(conns):,} connections -> {args.out}")


if __name__ == "__main__":
    main()
