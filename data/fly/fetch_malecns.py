#!/usr/bin/env python
"""Fetch the MaleCNS v1.0 connectivity and annotation tables (plan.md §1).

Two routes produce the same raw tables under data/fly/raw/:

  bulk (default)   the official flat-connectome Feather files from Janelia's public bucket
                   gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/ over HTTPS, md5-verified
                   against the bucket listing. No credentials needed.
  neuprint         the neuPrint API, dataset "male-cns:v1.0" (verified name, male-cns.janelia.org/download).
                   Needs NEUPRINT_TOKEN. Same content, but pages the whole graph through Cypher; use it only
                   where the bucket is unreachable.

    python data/fly/fetch_malecns.py                 # bulk
    python data/fly/fetch_malecns.py --neuprint      # API

Then: python data/fly/build_edges.py --region-column superclass
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
from pathlib import Path

BUCKET = "https://storage.googleapis.com/flyem-male-cns"
PREFIX = "v1.0/connectome-data/flat-connectome"
BULK_FILES = [  # the three tables build_edges.py needs (+ NT predictions for the §10 follow-up)
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
]


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def fetch_bulk(out_dir: Path):
    import requests

    out_dir.mkdir(parents=True, exist_ok=True)
    listing = requests.get(f"{BUCKET.replace('flyem-male-cns', 'storage/v1/b/flyem-male-cns')}/o",
                           params={"prefix": PREFIX + "/", "fields": "items(name,size,md5Hash)"}, timeout=60).json()
    expected = {Path(it["name"]).name: it["md5Hash"] for it in listing["items"]}
    for name in BULK_FILES:
        dst = out_dir / name
        if dst.exists() and md5_b64(dst) == expected[name]:
            print(f"ok      {name}")
            continue
        print(f"fetch   {name}")
        with requests.get(f"{BUCKET}/{PREFIX}/{name}", stream=True, timeout=1800) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
        got = md5_b64(dst)
        if got != expected[name]:
            sys.exit(f"md5 mismatch for {name}: {got} != {expected[name]}")
        print(f"ok      {name} (md5 verified)")
    (out_dir / "SOURCE.txt").write_text(
        f"gs://flyem-male-cns/{PREFIX}/  (MaleCNS v1.0 flat connectome, CC-BY 4.0)\n"
        + "".join(f"{expected[n]}  {n}\n" for n in BULK_FILES))


def fetch_neuprint(out_dir: Path, dataset: str, server: str, min_synapses: int):
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        sys.exit("set NEUPRINT_TOKEN (neuprint.janelia.org -> Account -> Auth Token)")
    from neuprint import Client, fetch_adjacencies, NeuronCriteria  # pip install neuprint-python

    c = Client(server, dataset=dataset, token=token)
    print(c.fetch_version())
    neurons, conns = fetch_adjacencies(NeuronCriteria(), NeuronCriteria(), min_total_weight=min_synapses)
    conns = conns.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    out_dir.mkdir(parents=True, exist_ok=True)
    conns.rename(columns={"bodyId_pre": "body_pre", "bodyId_post": "body_post"}).to_parquet(out_dir / "neuprint_connections.parquet", index=False)
    neurons.to_parquet(out_dir / "neuprint_neurons.parquet", index=False)
    print(f"{len(neurons):,} neurons, {len(conns):,} connections -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neuprint", action="store_true", help="use the neuPrint API instead of the bulk files")
    ap.add_argument("--dataset", default="male-cns:v1.0")
    ap.add_argument("--server", default="https://neuprint.janelia.org")
    ap.add_argument("--min-synapses", type=int, default=1)
    ap.add_argument("--out-dir", default="data/fly/raw")
    args = ap.parse_args()
    if args.neuprint:
        fetch_neuprint(Path(args.out_dir), args.dataset, args.server, args.min_synapses)
    else:
        fetch_bulk(Path(args.out_dir))


if __name__ == "__main__":
    main()
