#!/usr/bin/env python
"""Turn the raw MaleCNS tables into the two files the extractor reads.

    python data/fly/build_edges.py --region-column superclass

Outputs (gitignored, rebuildable):
    data/fly/neurons.parquet   body_id, type, instance, superclass, status, region
    data/fly/edges.parquet     src, dst, weight            (body ids, synapse count; self-loops dropped)

The `region` mapping is the one place a wrong choice turns "fly brain" into "fly spinal cord"
(README §3.1). In the v1.0 annotations the column that separates brain from nerve cord is
`superclass`: values are prefixed cb_ (central brain), vnc_ (ventral nerve cord), ol_ (optic lobe);
neurons that span compartments (ascending/descending, visual projection/centrifugal, ...) map to
'other' and are excluded from a central_brain pool. The full superclass -> region table is printed;
read it before running build_graph.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW = Path("data/fly/raw")
DEFAULT_WEIGHTS = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
DEFAULT_ANNOTATIONS = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

SUPERCLASS_PREFIX = {"cb_": "central_brain", "vnc_": "vnc", "ol_": "optic_lobe"}
# fallback for other annotation columns (neuPrint route, older releases)
BRAIN_TOKENS = ("brain", "central", "cb_", "cns_brain")
VNC_TOKENS = ("vnc", "ventral", "neuromere", "thoracic", "abdominal")


def region_from_superclass(value) -> str:
    if not isinstance(value, str):
        return "unannotated"
    for prefix, region in SUPERCLASS_PREFIX.items():
        if value.startswith(prefix):
            return region
    return "other"


def coarse_region(value) -> str:
    v = str(value).lower()
    if any(t in v for t in VNC_TOKENS):
        return "vnc"
    if any(t in v for t in BRAIN_TOKENS):
        return "central_brain"
    return "other"


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_feather(path) if path.suffix == ".feather" else pd.read_parquet(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="body_pre, body_post, weight table")
    ap.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS), help="bodyId + annotation columns")
    ap.add_argument("--region-column", default="superclass", help="annotation column to derive region from")
    ap.add_argument("--out-dir", default="data/fly")
    args = ap.parse_args()

    conns = read_table(Path(args.weights)).rename(columns={"bodyId_pre": "body_pre", "bodyId_post": "body_post"})
    conns = conns[conns["body_pre"] != conns["body_post"]]
    edges = conns.rename(columns={"body_pre": "src", "body_post": "dst"})[["src", "dst", "weight"]].reset_index(drop=True)
    edges.to_parquet(f"{args.out_dir}/edges.parquet", index=False)

    ann = read_table(Path(args.annotations)).rename(columns={"bodyId": "body_id"})
    keep = [c for c in ["body_id", "type", "instance", "superclass", "class", "status", "statusLabel"] if c in ann.columns]
    out = ann[keep].copy()
    col = args.region_column
    if col not in ann.columns:
        raise SystemExit(f"region column {col!r} not in annotations (columns: {list(ann.columns)})")
    mapper = region_from_superclass if col == "superclass" else coarse_region
    out["region"] = ann[col].map(mapper)
    out.to_parquet(f"{args.out_dir}/neurons.parquet", index=False)

    bodies = pd.concat([edges["src"], edges["dst"]]).unique()
    print(f"{len(bodies):,} bodies with connections, {len(edges):,} directed edges, {edges['weight'].sum():,} synaptic contacts")
    print(f"{len(out):,} annotated bodies; {(~pd.Series(bodies).isin(out['body_id'])).sum():,} connected bodies lack an annotation row")
    print(f"\n{col} -> region:")
    print(pd.crosstab(ann[col].fillna("<NaN>"), out["region"]).sort_values("central_brain" if "central_brain" in out["region"].values else out["region"].iloc[0], ascending=False).to_string())
    print("\nregion breakdown of annotated bodies:")
    print(out["region"].value_counts().to_string())
    connected = out[out["body_id"].isin(bodies)]
    print("\nregion breakdown of bodies with connections:")
    print(connected["region"].value_counts().to_string())


if __name__ == "__main__":
    main()
