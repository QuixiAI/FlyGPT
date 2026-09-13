#!/usr/bin/env python
"""Package the MaleCNS v1.0 traced connectome as a Hugging Face repo (safetensors + trust_remote_code).

    python data/fly/export_malecns_hf.py --out ~/malecns-hf [--name QuixiAI/MaleCNS-v1.0]

Nodes are every traced body that has at least one connection in the release's traced-only weight table;
edges are that table unchanged (self-loops kept, min synapses 1). Annotations are joined from the
body-annotations and body-neurotransmitters tables. Every number in the model card is computed here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.fly.build_edges import region_from_superclass  # noqa: E402

RAW = ROOT / "data/fly/raw"
FILES = {
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}
REGION_LABELS = ["central_brain", "optic_lobe", "vnc", "other", "unannotated"]
SUBSETS = {  # named neuron subsets by release superclass; "all" = every neuron
    "full_cns": "all",
    "central_brain": ["cb_intrinsic", "cb_sensory", "cb_motor", "cb_endocrine", "cb_efferent", "cb_sensory_tbc"],
    "optic_lobes": ["ol_intrinsic", "ol_sensory"],
    "vnc": ["vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent", "vnc_endocrine", "vnc_tbc", "vnc_sensory_tbc"],
    "cb_sensory": ["cb_sensory", "cb_sensory_tbc"],
    "visual_projection": ["visual_projection", "visual_projection_tbc"],
    "visual_centrifugal": ["visual_centrifugal"],
    "ascending": ["ascending_neuron", "sensory_ascending", "sensory_ascending_tbc", "efferent_ascending"],
    "descending": ["descending_neuron", "descending_neuron_tbc", "sensory_descending", "efferent_descending"],
}
NT_LABELS = ["unclear", "acetylcholine", "gaba", "glutamate", "dopamine", "histamine", "serotonin", "octopamine"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="MaleCNS-v1.0-connectome")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    md5 = dict(l.split()[::-1] for l in (RAW / "SOURCE.txt").read_text().splitlines()[1:] if l.strip())

    w = pd.read_feather(RAW / FILES["weights"], columns=["body_pre", "body_post", "weight"])
    node_id = np.unique(np.concatenate([w.body_pre.to_numpy(), w.body_post.to_numpy()]))
    src = np.searchsorted(node_id, w.body_pre.to_numpy()).astype(np.int32)
    dst = np.searchsorted(node_id, w.body_post.to_numpy()).astype(np.int32)
    count = w.weight.to_numpy().astype(np.int32)
    order = np.lexsort((dst, src))
    src, dst, count = src[order], dst[order], count[order]
    N, E = len(node_id), len(src)
    n_self = int((src == dst).sum())

    ann = pd.read_feather(RAW / FILES["annotations"]).rename(columns={"bodyId": "body_id"})
    ann = ann.set_index("body_id").reindex(node_id)
    superclass = ann["superclass"].fillna("unannotated")
    superclass_labels = sorted(superclass.unique().tolist())
    region = superclass.map(region_from_superclass).where(superclass != "unannotated", "unannotated")
    nt = pd.read_feather(RAW / FILES["neurotransmitters"], columns=["body", "consensus_nt", "predicted_nt_confidence"])
    nt = nt.drop_duplicates("body").set_index("body").reindex(node_id)
    nt_class = nt["consensus_nt"].fillna("unclear")
    assert set(nt_class.unique()) <= set(NT_LABELS), set(nt_class.unique()) - set(NT_LABELS)
    assert set(region.unique()) <= set(REGION_LABELS)

    tensors = {
        "graph.edge_index": torch.from_numpy(np.stack([src, dst])).contiguous(),
        "graph.synapse_count": torch.from_numpy(count).contiguous(),
        "graph.edge_weight_bf16": torch.from_numpy(count.astype(np.float32)).to(torch.bfloat16).contiguous(),
        "graph.node_id": torch.from_numpy(node_id.astype(np.int64)).contiguous(),
        "neuron.region": torch.from_numpy(region.map(REGION_LABELS.index).to_numpy().astype(np.int8)).contiguous(),
        "neuron.superclass": torch.from_numpy(superclass.map(superclass_labels.index).to_numpy().astype(np.int8)).contiguous(),
        "neuron.nt_class": torch.from_numpy(nt_class.map(NT_LABELS.index).to_numpy().astype(np.int8)).contiguous(),
        "neuron.nt_confidence": torch.from_numpy(nt["predicted_nt_confidence"].fillna(0).to_numpy().astype(np.float32)).to(torch.bfloat16).contiguous(),
    }
    save_file(tensors, out / "model.safetensors",
              metadata={"format": "pt", "source": "MaleCNS v1.0 flat connectome (CC-BY 4.0)", "architecture": "MaleCNSConnectome"})

    indeg = np.bincount(dst, minlength=N); outdeg = np.bincount(src, minlength=N)
    region_counts = region.value_counts().to_dict()
    thr = {t: int((count >= t).sum()) for t in (1, 2, 3, 5, 10)}
    stats = {
        "num_neurons": N, "num_edges": E, "num_self_loops": n_self, "synaptic_contacts": int(count.sum()),
        "synapse_count_min": int(count.min()), "synapse_count_max": int(count.max()),
        "edges_at_min_synapses": thr, "bf16_exact_fraction": float((count <= 256).mean()),
        "in_degree": {"mean": float(indeg.mean()), "max": int(indeg.max())},
        "out_degree": {"mean": float(outdeg.mean()), "max": int(outdeg.max())},
        "region_breakdown": {k: int(region_counts.get(k, 0)) for k in REGION_LABELS},
        "superclass_breakdown": {k: int(v) for k, v in superclass.value_counts().items()},
        "neurotransmitter_breakdown": {k: int(v) for k, v in nt_class.value_counts().items()},
    }
    # keep only superclasses that actually occur among connected traced neurons (e.g. *_tbc classes may not)
    subsets = {k: v if v == "all" else [c for c in v if c in superclass_labels] for k, v in SUBSETS.items()}
    stats["subset_sizes"] = {k: int(N if v == "all" else superclass.isin(v).sum()) for k, v in subsets.items()}
    config = {
        "model_type": "malecns", "architectures": ["MaleCNSConnectome"],
        "auto_map": {"AutoConfig": "configuration_malecns.MaleCNSConfig", "AutoModel": "modeling_malecns.MaleCNSConnectome"},
        "num_neurons": N, "num_edges": E, "num_self_loops": n_self, "min_synapses": 1, "release": "MaleCNS v1.0",
        "source_files": {k: {"file": v, "md5_base64": md5.get(v)} for k, v in FILES.items()},
        "region_labels": REGION_LABELS, "superclass_labels": superclass_labels, "nt_labels": NT_LABELS,
        "subsets": subsets, "stats": stats,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for f in ("configuration_malecns.py", "modeling_malecns.py"):
        shutil.copy(ROOT / "hf" / f, out / f)

    side = ann[[c for c in ["type", "instance", "superclass", "class", "status", "statusLabel", "somaSide", "somaNeuromere", "dimorphism", "fruDsx"] if c in ann.columns]].copy()
    side.insert(0, "body_id", node_id); side["region"] = region.to_numpy()
    side["consensus_nt"] = nt_class.to_numpy(); side["nt_confidence"] = nt["predicted_nt_confidence"].to_numpy()
    side["in_degree"] = indeg; side["out_degree"] = outdeg
    for c in side.columns:
        if side[c].dtype.name == "category" or side[c].dtype == object:
            side[c] = side[c].astype("string")
    side.reset_index(drop=True).to_parquet(out / "neurons.parquet", index=False)
    (out / "README.md").write_text(model_card(args.name, config, tensors))
    verify_lossless(out, w)
    print(json.dumps({"out": str(out), **{k: stats[k] for k in ("num_neurons", "num_edges", "num_self_loops", "synaptic_contacts", "region_breakdown", "edges_at_min_synapses", "bf16_exact_fraction")},
                      "safetensors_mb": round((out / "model.safetensors").stat().st_size / 1e6, 1)}, indent=2))


def verify_lossless(out: Path, w: pd.DataFrame):
    """Reload the safetensors and check the edge table equals the release table exactly."""
    from safetensors.torch import load_file
    t = load_file(out / "model.safetensors")
    ids = t["graph.node_id"].numpy()
    got = pd.DataFrame({"body_pre": ids[t["graph.edge_index"][0].numpy()], "body_post": ids[t["graph.edge_index"][1].numpy()],
                        "weight": t["graph.synapse_count"].numpy().astype(np.int64)})
    key = ["body_pre", "body_post"]
    a = got.sort_values(key).reset_index(drop=True)
    b = w[["body_pre", "body_post", "weight"]].astype(np.int64).sort_values(key).reset_index(drop=True)
    assert len(a) == len(b) and (a.to_numpy() == b.to_numpy()).all(), "lossless check FAILED"
    print(f"lossless check passed: {len(a):,} edges identical to the release table")


def model_card(name: str, config: dict, tensors: dict) -> str:
    s = config["stats"]
    rows = "\n".join(f"| `{k}` | {tuple(v.shape)} | {str(v.dtype).replace('torch.', '')} | {v.numel() * v.element_size() / 1e6:.1f} MB |"
                     for k, v in tensors.items())
    regions = "\n".join(f"| {k} | {v:,} |" for k, v in s["region_breakdown"].items())
    thr = " · ".join(f"≥{t}: {n:,}" for t, n in s["edges_at_min_synapses"].items())
    nts = "\n".join(f"| {k} | {v:,} |" for k, v in s["neurotransmitter_breakdown"].items())
    files = "\n".join(f"| `{v['file']}` | `{v['md5_base64']}` |" for v in config["source_files"].values())
    subsets = "\n".join(f"| `{k}` | {s['subset_sizes'][k]:,} | {'all neurons' if v == 'all' else ', '.join(v)} |" for k, v in config["subsets"].items())
    return f"""---
license: cc-by-4.0
library_name: transformers
tags: [connectome, fruit-fly, drosophila, malecns, neuroscience, graph, custom_code]
---

# {name}

The **MaleCNS v1.0** connectome, the complete wiring diagram of an adult male *Drosophila melanogaster* central
nervous system (brain, optic lobes, and ventral nerve cord), packaged as Hugging Face `safetensors` so it loads as
PyTorch tensors with `AutoModel.from_pretrained(..., trust_remote_code=True)`.

> **`{name}` is a lossless Safetensors packaging of the MaleCNS connectivity tables. It does not impose a neuron
> model, neurotransmitter sign convention, weight normalization, or language-model architecture.**

These are the **original fly weights**: every directed neuron-to-neuron connection in the release's traced-neuron
connectivity table with its exact synapse count. Nothing here is learned, trained, scaled, signed, or simulated.
The build script re-reads the saved tensors and checks them against the release table edge by edge.
Subsets (central brain, optic lobes, nerve cord, sensory, ascending, descending, ...) are provided as masks so that
downstream users, not this repository, decide what "the model" is.

| | |
|---|---|
| Neurons (traced bodies with ≥ 1 connection) | {s['num_neurons']:,} |
| Directed connections | {s['num_edges']:,} (of which {s['num_self_loops']:,} autapses, kept) |
| Synaptic contacts represented | {s['synaptic_contacts']:,} |
| Synapse count per connection | {s['synapse_count_min']} – {s['synapse_count_max']:,} |
| Connections at a minimum synapse count | {thr} |
| Mean in/out degree | {s['in_degree']['mean']:.1f} (max in {s['in_degree']['max']:,}, max out {s['out_degree']['max']:,}) |

## Region breakdown

Coarse region from the release's `superclass` annotation: `cb_*` → central brain, `ol_*` → optic lobe, `vnc_*` →
ventral nerve cord; neurons spanning compartments (ascending, descending, visual projection/centrifugal, ...) → other.

| region | neurons |
|---|--:|
{regions}

## Named subsets (`cns.subset_mask(name)`)

| subset | neurons | superclasses |
|---|--:|---|
{subsets}

## Consensus neurotransmitter (per neuron)

| neurotransmitter | neurons |
|---|--:|
{nts}

## What is in `model.safetensors`

| tensor | shape | dtype | size |
|---|---|---|---|
{rows}

`graph.edge_index` is `(source, destination)` in contiguous neuron indices; `graph.node_id` maps an index back to the
MaleCNS body id. `graph.synapse_count` is the canonical, exact count. `graph.edge_weight_bf16` is a derived
convenience copy for bf16 pipelines: exact for counts up to 256 ({100 * s['bf16_exact_fraction']:.2f}% of connections),
rounded to 8 significant bits above. Label indices in `neuron.*` resolve through `config.region_labels`, `config.superclass_labels`,
and `config.nt_labels`. `neurons.parquet` carries the full per-neuron annotation strings (type, instance,
superclass, class, status, side, neuromere, dimorphism, fru/dsx, neurotransmitter, degrees).

## How this differs from `ngxson/fly-llm-hf`

MaleCNS is already on the Hub in [ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf), so this is not the
first packaging. It is a different layer of the stack: that repository is a **derived language model**; this one is
the **source connectome**, unmodified. Facts about fly-llm-hf below are taken from its model card.

| | this repository | `ngxson/fly-llm-hf` |
|---|---|---|
| Purpose | package MaleCNS losslessly for PyTorch / graph ML | a toy language model (echo-state reservoir) |
| Source | official flat-connectome files from the Janelia bucket, md5-verified | MaleCNS as packaged by the Xenova fruit-fly-simulation space |
| Scope | full CNS: {s['num_neurons']:,} traced neurons, {s['num_edges']:,} connections, subsets as masks | central brain only (cb_sensory, visual_projection, cb_intrinsic, ascending, descending): 49,393 neurons, 9,050,172 edges |
| Edge values | exact synapse counts (int32); bf16 copy clearly marked as derived | signed synapse counts (ACh +1, GABA/glutamate −1, others 0), globally rescaled to spectral radius 0.99, plus a learned per-neuron gain |
| Neurotransmitter data | separate per-neuron label + confidence tensors, not applied to the graph | used to assign the sign of every outgoing edge |
| Dynamics | none (`forward` is one linear propagation step, for convenience) | leaky-tanh reservoir, `a = 0.9`, 8-token delay line into 14,069 sensory neurons |
| Trainable parameters | none | input projection, per-neuron gains, LayerNorm + readout (52.8M); the connectome is frozen |
| Tokenizer / task | none | byte-level BPE (1024) / TinyStories |
| `transformers` role | `AutoModel` returning the graph | `AutoModelForCausalLM` generating text |

```text
Janelia MaleCNS v1.0 ──► this repository (lossless tensors) ──► FlyGPT (trains the fly's edge values on Shakespeare)
Janelia MaleCNS v1.0 ──► Xenova packaging ──► fly-llm-hf (signed, rescaled, frozen reservoir + TinyStories readout)
```

## Usage

```python
import torch
from transformers import AutoModel

cns = AutoModel.from_pretrained("{name}", trust_remote_code=True)

W = cns.sparse_weight()                          # sparse COO [N, N], rows = destination, exact counts, fp32
W_cb = cns.sparse_weight(nodes=cns.subset_mask("central_brain"), min_synapses=3)   # induced subgraph, relabelled
sub = cns.subgraph(cns.superclass_mask("cb_intrinsic"))                             # plain tensors + body ids

x = torch.zeros(1, cns.num_neurons); x[0, 0] = 1.0
incoming = cns(x)                                # one propagation step: synapse-weighted input to every neuron
```

## Provenance

Built by [`data/fly/export_malecns_hf.py`](https://github.com/QuixiAI/FlyGPT) from these files in
`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, md5-verified against the bucket listing:

| file | md5 (base64) |
|---|---|
{files}

The traced-only weight table is the neuron-level graph (the full table also contains ~88M unproofread segment
fragments). Neurotransmitter labels are the release's per-neuron `consensus_nt` with its prediction confidence.

## License and citation

Released under **CC-BY 4.0** by the FlyEM Project Team (HHMI Janelia), the University of Cambridge (Dept. of Zoology),
the MRC Laboratory of Molecular Biology, and Google Research. This repository is a repackaging and carries the same
license; attribute the original authors:

> Berg, S., Beckett, I. R., Costa, M., Schlegel, P., Januszewski, M., Marin, E. C., Nern, A., Preibisch, S., et al. (2026).
> Sexual dimorphism in the complete *Drosophila* male central nervous system connectome. *Cell* 189, 5504–5526.e15.
> https://doi.org/10.1016/j.cell.2026.08.015 (preprint: https://doi.org/10.1101/2025.10.09.680999)

Official site: https://male-cns.janelia.org · neuPrint dataset `male-cns:v1.0`.

This repository is the data backbone of [FlyGPT](https://github.com/QuixiAI/FlyGPT), which trains a language model
whose recurrent architecture is a subgraph of this connectome.
"""


if __name__ == "__main__":
    main()
