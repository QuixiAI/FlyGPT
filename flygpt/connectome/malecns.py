"""MaleCNS v1.0 access (README §1).

Two raw tables, produced by data/fly/fetch_malecns.py:
    data/fly/raw_neurons.parquet      bodyId, type, instance, and whatever region/annotation columns exist
    data/fly/raw_connections.parquet  bodyId_pre, bodyId_post, weight (synapse count)

and the two derived files this module reads:
    data/fly/neurons.parquet  body_id, type, instance, region
    data/fly/edges.parquet    src, dst, weight

TODO before anything ships: record the MaleCNS v1.0 license and citation in README.md.
TODO: confirm which annotation column distinguishes central brain from VNC in the v1.0
release, and encode the mapping in `region_of()`. Until then, `central_brain` filtering
raises so a wrong pool cannot silently become "the fly brain".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import GraphConfig

REGION_COLUMN_CANDIDATES = ["region", "superclass", "class", "somaNeuromere", "rootSide"]


def region_of(neurons: pd.DataFrame) -> pd.Series:
    """Map each neuron to a coarse region label: 'central_brain' | 'vnc' | 'other'."""
    if "region" in neurons.columns:
        return neurons["region"].fillna("unknown")
    raise NotImplementedError(
        "neurons.parquet has no 'region' column. Populate it in data/fly/build_edges.py from the v1.0 "
        f"annotations (candidates: {REGION_COLUMN_CANDIDATES}) before running a central_brain extraction."
    )


def load_candidate_pool(cfg: GraphConfig):
    """Returns (src_ids, dst_ids, weight, candidate_ids, regions: dict body_id -> region)."""
    edges = pd.read_parquet(cfg.edges_path)
    neurons = pd.read_parquet(cfg.neurons_path)
    regions = dict(zip(neurons["body_id"].astype(int), region_of(neurons)))
    if cfg.region_filter in ("central_brain", "vnc"):
        cand = neurons.loc[[r == cfg.region_filter for r in regions.values()], "body_id"].to_numpy(np.int64)
    elif cfg.region_filter in ("whole_cns", "none"):
        cand = neurons["body_id"].to_numpy(np.int64)
    else:
        raise ValueError(f"unknown region_filter {cfg.region_filter!r}")
    return (edges["src"].to_numpy(np.int64), edges["dst"].to_numpy(np.int64),
            edges["weight"].to_numpy(np.float32), cand, regions)
