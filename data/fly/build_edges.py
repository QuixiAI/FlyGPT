#!/usr/bin/env python
"""Turn the MaleCNS connectome into the two files the extractor reads.

    python data/fly/build_edges.py                        # from the Hub: QuixiAI/MaleCNS, pinned revision (default)
    python data/fly/build_edges.py --source raw           # from the release Feather files in data/fly/raw/

The foundation is the lossless packaging at https://huggingface.co/QuixiAI/MaleCNS (built from the official
release by data/fly/export_malecns_hf.py). Both routes produce identical files; the raw route exists only to
rebuild that repository. Writes data/fly/PROVENANCE.json (repo, revision, row counts) next to the outputs.

Outputs (gitignored, rebuildable):
    data/fly/neurons.parquet   body_id, type, instance, superclass, status, region
    data/fly/edges.parquet     src, dst, weight            (body ids, synapse count; self-loops dropped)

The `region` mapping is the one place a wrong choice turns "fly brain" into "fly spinal cord"
(plan.md §3.1). In the v1.0 annotations the column that separates brain from nerve cord is
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
HF_REPO = "QuixiAI/MaleCNS"
HF_REVISION = "6330780a4c8ab2519ce1c19b64aefced59581085"   # pinned; bump deliberately and log it in notes/decisions.md
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


def load_from_hub(repo: str, revision: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(connections, annotations, provenance) from the lossless Hub packaging."""
    import numpy as np
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    st = hf_hub_download(repo, "model.safetensors", revision=revision)
    t = load_file(st)
    ids = t["neuron_id"]
    conns = pd.DataFrame({"body_pre": ids[t["edge_src"]], "body_post": ids[t["edge_dst"]], "weight": t["synapse_count"].astype(np.int64)})
    ann = pd.read_parquet(hf_hub_download(repo, "neurons.parquet", revision=revision))
    prov = {"source": f"hf:{repo}", "revision": revision, "bodies": int(len(ids)), "edges": int(len(conns)),
            "annotation_rows": int(len(ann))}
    return conns, ann, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "raw"], default="hf", help="hf: QuixiAI/MaleCNS (default); raw: release Feather files")
    ap.add_argument("--revision", default=HF_REVISION, help="Hub revision to pin (hf source)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="body_pre, body_post, weight table (raw source)")
    ap.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS), help="bodyId + annotation columns")
    ap.add_argument("--region-column", default="superclass", help="annotation column to derive region from")
    ap.add_argument("--out-dir", default="data/fly")
    args = ap.parse_args()

    if args.source == "hf":
        conns, ann, prov = load_from_hub(HF_REPO, args.revision)
    else:
        conns = read_table(Path(args.weights)).rename(columns={"bodyId_pre": "body_pre", "bodyId_post": "body_post"})
        ann = read_table(Path(args.annotations)).rename(columns={"bodyId": "body_id"})
        prov = {"source": f"raw:{Path(args.weights).name}", "annotations": Path(args.annotations).name, "edges": int(len(conns))}
    conns = conns[conns["body_pre"] != conns["body_post"]]
    edges = conns.rename(columns={"body_pre": "src", "body_post": "dst"})[["src", "dst", "weight"]].reset_index(drop=True)
    edges.to_parquet(f"{args.out_dir}/edges.parquet", index=False)

    keep = [c for c in ["body_id", "type", "instance", "superclass", "class", "status", "statusLabel"] if c in ann.columns]
    out = ann[keep].copy()
    col = args.region_column
    if col not in ann.columns:
        raise SystemExit(f"region column {col!r} not in annotations (columns: {list(ann.columns)})")
    mapper = region_from_superclass if col == "superclass" else coarse_region
    out["region"] = ann[col].map(mapper)
    out.to_parquet(f"{args.out_dir}/neurons.parquet", index=False)
    import json
    prov.update(region_column=col, self_loops_dropped=True, directed_edges=int(len(edges)))
    Path(args.out_dir, "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    print("provenance:", prov)

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
