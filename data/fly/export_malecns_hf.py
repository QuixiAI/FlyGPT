#!/usr/bin/env python
"""Package the MaleCNS v1.0 connectivity tables as a lossless Hugging Face repo (safetensors + trust_remote_code).

    python data/fly/export_malecns_hf.py --out ~/malecns-hf [--name QuixiAI/malecns]

Canonical source: the release's full segment-to-segment table `connectome-weights-male-cns-v1.0-minconf-0.5.feather`
(body_pre, body_post, weight). It is mirrored in file row order with the `weight` column untouched as
`synapse_count`. Annotations and the neurotransmitter table are mirrored as separate label tensors and verbatim
parquet sidecars. Nothing is signed, scaled, normalized, rounded, or initialized. After writing, the script reloads
the safetensors and checks (a) the edge table equals the source row by row and (b) restricting to bodies with
status == Traced reproduces the release's traced-only table exactly.
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
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/fly/raw"
FILES = {
    "connectivity": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    "connectivity_traced_only": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}
STATUS_LABELS = ["Traced", "Orphan", "Glia", "Unimportant", "Assign", "Anchor", "unannotated"]
NT_LABELS = ["unclear", "acetylcholine", "gaba", "glutamate", "dopamine", "histamine", "serotonin", "octopamine"]
SUBSETS = {  # documented convenience: lists of release superclasses. "all" = every body in the table.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="MaleCNS-v1.0-connectome")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    md5 = dict(l.split()[::-1] for l in (RAW / "SOURCE.txt").read_text().splitlines()[1:] if l.strip())

    # ---- connectivity, literal ---------------------------------------------------------------------
    w = pd.read_feather(RAW / FILES["connectivity"], columns=["body_pre", "body_post", "weight"])
    neuron_id = np.unique(np.concatenate([w.body_pre.to_numpy(), w.body_post.to_numpy()]))
    edge_src = np.searchsorted(neuron_id, w.body_pre.to_numpy()).astype(np.int32)
    edge_dst = np.searchsorted(neuron_id, w.body_post.to_numpy()).astype(np.int32)
    synapse_count = w.weight.to_numpy().astype(np.int32)
    assert (synapse_count == w.weight.to_numpy()).all(), "weight column does not fit int32"
    N, E = len(neuron_id), len(edge_src)

    # ---- annotations -> label tensors (release vocabularies only) ---------------------------------
    ann = pd.read_feather(RAW / FILES["annotations"]).rename(columns={"bodyId": "body_id"})
    ann_n = ann.set_index("body_id").reindex(neuron_id)
    status = ann_n["status"].astype("string").fillna("unannotated")
    status = status.where(status.isin(STATUS_LABELS), "unannotated")
    superclass = ann_n["superclass"].astype("string").fillna("unannotated")
    superclass_labels = sorted(superclass.unique().tolist())
    subsets = {k: v if v == "all" else [c for c in v if c in superclass_labels] for k, v in SUBSETS.items()}

    # ---- neurotransmitter table, mirrored as its own group ---------------------------------------
    nt = pd.read_feather(RAW / FILES["neurotransmitters"], columns=["body", "consensus_nt", "predicted_nt_confidence"])
    nt = nt[nt.body.isin(neuron_id)].reset_index(drop=True)
    nt_index = np.searchsorted(neuron_id, nt.body.to_numpy()).astype(np.int32)
    nt_class = nt["consensus_nt"].astype("string").fillna("unclear")
    assert set(nt_class.unique()) <= set(NT_LABELS), set(nt_class.unique()) - set(NT_LABELS)
    M = len(nt)

    tensors = {
        "neuron_id": torch.from_numpy(neuron_id.astype(np.int64)),
        "edge_src": torch.from_numpy(edge_src),
        "edge_dst": torch.from_numpy(edge_dst),
        "synapse_count": torch.from_numpy(synapse_count),
        "neuron_status": torch.from_numpy(status.map(STATUS_LABELS.index).to_numpy().astype(np.int8)),
        "neuron_superclass": torch.from_numpy(superclass.map(superclass_labels.index).to_numpy().astype(np.int8)),
        "nt_neuron_index": torch.from_numpy(nt_index),
        "nt_class": torch.from_numpy(nt_class.map(NT_LABELS.index).to_numpy().astype(np.int8)),
        "nt_confidence": torch.from_numpy(nt["predicted_nt_confidence"].to_numpy().astype(np.float64)),
    }
    tensors = {k: v.contiguous() for k, v in tensors.items()}
    save_file(tensors, out / "model.safetensors",
              metadata={"format": "pt", "source": "MaleCNS v1.0 flat connectome (CC-BY 4.0)", "architecture": "MaleCNSConnectome"})

    # ---- stats (every number in the card comes from here) -----------------------------------------
    indeg, outdeg = np.bincount(edge_dst, minlength=N), np.bincount(edge_src, minlength=N)
    traced = (status == "Traced").to_numpy()
    tt = traced[edge_src] & traced[edge_dst]
    stats = {
        "num_neurons": N, "num_edges": E, "num_self_loops": int((edge_src == edge_dst).sum()),
        "synaptic_contacts": int(synapse_count.sum()),
        "synapse_count_min": int(synapse_count.min()), "synapse_count_max": int(synapse_count.max()),
        "edges_at_min_synapses": {t: int((synapse_count >= t).sum()) for t in (1, 2, 3, 5, 10)},
        "in_degree": {"mean": float(indeg.mean()), "max": int(indeg.max())},
        "out_degree": {"mean": float(outdeg.mean()), "max": int(outdeg.max())},
        "status_breakdown": {k: int(v) for k, v in status.value_counts().items()},
        "traced_subgraph": {"neurons": int(traced.sum()), "edges": int(tt.sum()), "synaptic_contacts": int(synapse_count[tt].sum())},
        "superclass_breakdown": {k: int(v) for k, v in superclass.value_counts().items()},
        "subset_sizes": {k: int(N if v == "all" else superclass.isin(v).sum()) for k, v in subsets.items()},
        "neurotransmitter_rows": M,
        "neurotransmitter_breakdown": {k: int(v) for k, v in nt_class.value_counts().items()},
    }
    config = {
        "model_type": "malecns", "architectures": ["MaleCNSConnectome"],
        "auto_map": {"AutoConfig": "configuration_malecns.MaleCNSConfig", "AutoModel": "modeling_malecns.MaleCNSConnectome"},
        "num_neurons": N, "num_edges": E, "num_nt_rows": M, "release": "MaleCNS v1.0",
        "source_files": {k: {"file": v, "md5_base64": md5.get(v)} for k, v in FILES.items()},
        "status_labels": STATUS_LABELS, "superclass_labels": superclass_labels, "nt_labels": NT_LABELS,
        "subsets": subsets, "stats": stats,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    for f in ("configuration_malecns.py", "modeling_malecns.py"):
        shutil.copy(ROOT / "hf" / f, out / f)

    # ---- verbatim sidecars ---------------------------------------------------------------------------
    side = ann[ann.body_id.isin(neuron_id)].reset_index(drop=True)
    for c in side.columns:
        if side[c].dtype == object or side[c].dtype.name == "category":
            side[c] = side[c].map(lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else (None if x is None else str(x))).astype("string")
    side.to_parquet(out / "neurons.parquet", index=False)
    ntv = pd.read_feather(RAW / FILES["neurotransmitters"])
    ntv = ntv[ntv.body.isin(neuron_id)].reset_index(drop=True)
    for c in ntv.columns:
        if ntv[c].dtype == object:
            ntv[c] = ntv[c].astype("string")
    ntv.to_parquet(out / "neurotransmitters.parquet", index=False)

    (out / "README.md").write_text(model_card(args.name, config, tensors))
    verify(out, w)
    print(json.dumps({"out": str(out), **{k: stats[k] for k in ("num_neurons", "num_edges", "num_self_loops", "synaptic_contacts", "traced_subgraph", "neurotransmitter_rows")},
                      "safetensors_mb": round((out / "model.safetensors").stat().st_size / 1e6, 1)}, indent=2))


def verify(out: Path, w: pd.DataFrame):
    t = load_file(out / "model.safetensors")
    ids = t["neuron_id"].numpy()
    src, dst, cnt = ids[t["edge_src"].numpy()], ids[t["edge_dst"].numpy()], t["synapse_count"].numpy()
    assert (src == w.body_pre.to_numpy()).all() and (dst == w.body_post.to_numpy()).all() and (cnt == w.weight.to_numpy()).all()
    print(f"lossless check passed: {len(cnt):,} edges identical to the release table, in file row order")
    labels = json.loads((out / "config.json").read_text())["status_labels"]
    traced = t["neuron_status"].numpy() == labels.index("Traced")
    keep = traced[t["edge_src"].numpy()] & traced[t["edge_dst"].numpy()]
    a = pd.DataFrame({"body_pre": src[keep], "body_post": dst[keep], "weight": cnt[keep]}).sort_values(["body_pre", "body_post"]).to_numpy()
    b = pd.read_feather(RAW / FILES["connectivity_traced_only"], columns=["body_pre", "body_post", "weight"]).sort_values(["body_pre", "body_post"]).to_numpy()
    assert a.shape == b.shape and (a == b).all(), "status==Traced restriction != release traced-only table"
    print(f"subset check passed: status==Traced reproduces the release's traced-only table ({len(b):,} edges)")


def model_card(name: str, config: dict, tensors: dict) -> str:
    s = config["stats"]
    rows = "\n".join(f"| `{k}` | {tuple(v.shape)} | {str(v.dtype).replace('torch.', '')} | {v.numel() * v.element_size() / 1e6:.1f} MB |"
                     for k, v in tensors.items())
    statuses = "\n".join(f"| {k} | {v:,} |" for k, v in s["status_breakdown"].items())
    subsets = "\n".join(f"| `{k}` | {s['subset_sizes'][k]:,} | {'every body in the table' if v == 'all' else ', '.join(v)} |" for k, v in config["subsets"].items())
    nts = "\n".join(f"| {k} | {v:,} |" for k, v in s["neurotransmitter_breakdown"].items())
    files = "\n".join(f"| {k} | `{v['file']}` | `{v['md5_base64']}` |" for k, v in config["source_files"].items())
    thr = " · ".join(f"≥{t}: {n:,}" for t, n in s["edges_at_min_synapses"].items())
    ts = s["traced_subgraph"]
    return f"""---
license: cc-by-4.0
library_name: transformers
tags: [connectome, fruit-fly, drosophila, malecns, neuroscience, graph, custom_code]
---

# {name}

The **MaleCNS v1.0** connectome, the complete wiring diagram of an adult male *Drosophila melanogaster* central
nervous system (brain, optic lobes, and ventral nerve cord), packaged as Hugging Face `safetensors` so it loads as
PyTorch tensors with `AutoModel.from_pretrained(..., trust_remote_code=True)`.

**Why this exists.** MaleCNS reached the Hub through [ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf),
a fun demonstration that the fly's central brain can serve as a frozen echo-state reservoir for a TinyStories language
model. That repository is a *model*: it keeps 49,393 central-brain neurons, assigns each synapse a sign from the
presynaptic neuron's predicted neurotransmitter, rescales the whole matrix to a spectral radius of 0.99, and trains
projections, per-neuron gains, and a readout around it. Those are good modeling decisions for that project, but
they are baked into its weights, so anyone who wants the connectome for something else has to undo them or go back
to the raw release. This repository is the layer underneath: the **source connectome, unmodified**, so that
fly-llm-hf, [FlyGPT](https://github.com/QuixiAI/FlyGPT), graph-ML work, and simulations can all start from the same
lossless tensors. At a high level the differences are:

- **Whole nervous system, not a subset.** Every body in the release's connectivity file: brain, optic lobes, and
  ventral nerve cord, with subsets available as masks rather than chosen for you.
- **Raw anatomical counts, not neural weights.** `synapse_count` is the release's `weight` column, untouched: no
  sign convention, no spectral rescaling, no normalization, no bf16 rounding, no random initialization.
- **Neurotransmitter predictions kept separate.** Mirrored as their own tensors so a sign convention is something
  you apply, not something you inherit.
- **Verified lossless.** The build reloads the saved tensors and checks them against the release table edge by edge.

> **A lossless packaging of the MaleCNS connectivity tables. It imposes no neuron model, no neurotransmitter sign
> convention, no normalization, no rounding, no initialization, and no language-model architecture.**

The canonical content is the release's full segment-to-segment connectivity file, mirrored literally: for every row
`body_pre, body_post, weight` there is one edge `edge_src, edge_dst, synapse_count`, in the file's row order, with the
`weight` column untouched. These values are **anatomical connection weights**: the number of detected synaptic
contacts from one body to another. They are not physiological synaptic efficacies. Any use of them as neural-network
weights (sign, scale, normalization, random re-initialization) is a downstream modeling choice and belongs downstream.

| | |
|---|---|
| Bodies (every segment with a synapse in the table) | {s['num_neurons']:,} |
| Directed connections | {s['num_edges']:,} (of which {s['num_self_loops']:,} autapses, kept) |
| Synaptic contacts represented | {s['synaptic_contacts']:,} |
| Synapse count per connection | {s['synapse_count_min']} – {s['synapse_count_max']:,} |
| Connections at a minimum synapse count | {thr} |
| Mean in/out degree | {s['in_degree']['mean']:.1f} (max in {s['in_degree']['max']:,}, max out {s['out_degree']['max']:,}) |
| Proofread neurons (`status == Traced`) | {ts['neurons']:,} bodies, {ts['edges']:,} connections between them, {ts['synaptic_contacts']:,} contacts |

The release also ships a traced-only connectivity file. It is not packaged separately because it is exactly this table
restricted to bodies with `status == Traced`; the build script checks that equality edge by edge.

## Side by side with `ngxson/fly-llm-hf`

Facts about fly-llm-hf are taken from its model card.

| | this repository | `ngxson/fly-llm-hf` |
|---|---|---|
| Purpose | package MaleCNS losslessly for PyTorch / graph ML | a toy language model (echo-state reservoir) |
| Source | official flat-connectome files from the Janelia bucket, md5-verified | MaleCNS as packaged by the Xenova fruit-fly-simulation space |
| Scope | the full table: {s['num_neurons']:,} bodies, {s['num_edges']:,} connections; subsets as masks | central brain only (cb_sensory, visual_projection, cb_intrinsic, ascending, descending): 49,393 neurons, 9,050,172 edges |
| Edge values | exact synapse counts (int32), file row order, untouched | signed synapse counts (ACh +1, GABA/glutamate −1, others 0), globally rescaled to spectral radius 0.99, plus a learned per-neuron gain |
| Neurotransmitter data | the release table mirrored as separate tensors, not applied to the graph | used to assign the sign of every outgoing edge |
| Dynamics | none (`forward` is one linear propagation step, for convenience) | leaky-tanh reservoir, `a = 0.9`, 8-token delay line into 14,069 sensory neurons |
| Trainable parameters | none | input projection, per-neuron gains, LayerNorm + readout (52.8M); the connectome is frozen |
| Tokenizer / task | none | byte-level BPE (1024) / TinyStories |
| `transformers` role | `AutoModel` returning the graph | `AutoModelForCausalLM` generating text |

```text
Janelia MaleCNS v1.0 ──► this repository (synapse counts, untouched) ──► FlyGPT (its own trainable edge weights on the same topology)
Janelia MaleCNS v1.0 ──► Xenova packaging ──► fly-llm-hf (signed, rescaled, frozen reservoir + TinyStories readout)
```

## What is in `model.safetensors`

| tensor | shape | dtype | size |
|---|---|---|---|
{rows}

`neuron_id` maps a neuron index to its MaleCNS body id. `neuron_status` and `neuron_superclass` are the release's
own annotation vocabularies (`config.status_labels`, `config.superclass_labels`). The neurotransmitter table is
mirrored as its own group (`nt_*`, one row per body that has a prediction), because Janelia publishes it
independently of the connectivity weights: `nt_class` is the table's `consensus_nt`, `nt_confidence` its
`predicted_nt_confidence` in float64. Full string columns of both source tables are in `neurons.parquet` and
`neurotransmitters.parquet`, verbatim.

## Proofreading status (`neuron_status`)

| status | bodies |
|---|--:|
{statuses}

## Named subsets (`cns.subset_mask(name)`)

Documented convenience only: each subset is a list of release `superclass` values. Nothing in the data depends on it.

| subset | bodies | superclasses |
|---|--:|---|
{subsets}

## Consensus neurotransmitter (rows of the neurotransmitter table)

| neurotransmitter | bodies |
|---|--:|
{nts}

## Usage

```python
import torch
from transformers import AutoModel

cns = AutoModel.from_pretrained("{name}", trust_remote_code=True)

neurons = cns.status_mask("Traced")                              # proofread neurons only
W = cns.sparse_weight(nodes=neurons)                             # sparse COO, rows = destination, raw counts
W_cb = cns.sparse_weight(nodes=neurons & cns.subset_mask("central_brain"), min_synapses=3)
sub = cns.subgraph(neurons & cns.superclass_mask("cb_intrinsic"))   # edge_src/edge_dst/synapse_count/neuron_id
nt_class, nt_conf = cns.neuron_nt()                              # per-neuron consensus neurotransmitter

x = torch.zeros(1, cns.num_neurons); x[0, 0] = 1.0
incoming = cns(x)                                                # one propagation step with raw counts
```

## Provenance

Built by [`data/fly/export_malecns_hf.py`](https://github.com/QuixiAI/FlyGPT) from these files in
`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`, md5-verified against the bucket listing:

| role | file | md5 (base64) |
|---|---|---|
{files}

## Citation

If you use this repository, please cite it **and** the MaleCNS dataset paper it repackages.

This repository:

```bibtex
@misc{{hartford2026malecns,
  title        = {{MaleCNS: the MaleCNS v1.0 fruit-fly connectome as lossless safetensors}},
  author       = {{Hartford, Eric}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/QuixiAI/MaleCNS}}}},
  note         = {{Repackaging of Berg et al. (2026), CC-BY 4.0}}
}}
```

The dataset (required by the CC-BY 4.0 license):

> Berg, S., Beckett, I. R., Costa, M., Schlegel, P., Januszewski, M., Marin, E. C., Nern, A., Preibisch, S., et al. (2026).
> Sexual dimorphism in the complete *Drosophila* male central nervous system connectome. *Cell* 189, 5504–5526.e15.
> https://doi.org/10.1016/j.cell.2026.08.015 (preprint: https://doi.org/10.1101/2025.10.09.680999)

```bibtex
@article{{berg2026malecns,
  title     = {{Sexual dimorphism in the complete {{Drosophila}} male central nervous system connectome}},
  author    = {{Berg, Stuart and Beckett, Isabella R. and Costa, Marta and Schlegel, Philipp and Januszewski, Michał and Marin, Elizabeth C. and Nern, Aljoscha and Preibisch, Stephan and Qiu, Wei and Takemura, Shin-ya and Fragniere, Alexandra M.C. and Champion, Andrew S. and Adjavon, Diane-Yayra and Cook, Michael and Gkantia, Marina and Hayworth, Kenneth J. and Huang, Gary B. and Katz, William T. and Kämpf, Florian and Lu, Zhiyuan and Ordish, Christopher and Paterson, Tyler and Stürner, Tomke and Trautman, Eric T. and Whittle, Catherine R. and Burnett, Laura E. and Hoeller, Judith and Li, Feng and Loesche, Frank and Morris, Billy J. and Pietzsch, Tobias and Pleijzier, Markus W. and Silva, Valeria and Yin, Yijie and Ali, Iris and Badalamente, Griffin and Bates, Alexander Shakeel and Beresford, Rory J. and Bogovic, John and Brooks, Paul and Cachero, Sebastian and Canino, Brandon S. and Chaisrisawatsuk, Bhumpanya and Clements, Jody and Crowe, Arthur and de Haan Vicente, Inês and Dempsey, Georgia and Donà, Erika and Dos Santos, Márcia and Dreher, Marisa and Dunne, Christopher R. and Eichler, Katharina and Finley-May, Samantha and Flynn, Miriam A. and Hameed, Imran and Hopkins, Gary Patrick and Hubbard, Philip M. and Kiassat, Ladann and Kovalyak, Julie and Lauchie, Shirley A. and Leonard, Meghan and Lohff, Alanna and Longden, Kit D. and Maldonado, Charli A. and Moitra, Ilina and Moon, Sung Soo and Mooney, Caroline and Munnelly, Eva J. and Okeoma, Nneoma and Olbris, Donald J. and Pai, Anika and Patel, Birava and Phillips, Emily M. and Plaza, Stephen M. and Richards, Alana and Rivas Salinas, Jennifer and Roberts, Ruairí J.V. and Rogers, Edward M. and Scott, Ashley L. and Scuderi, Louis A. and Seenivasan, Pavithraa and Serratosa Capdevila, Laia and Smith, Claire and Svirskas, Rob and Takemura, Satoko and Tastekin, Ibrahim and Thomson, Alexander and Umayam, Lowell and Walsh, John J. and Whittome, Holly and Xu, C. Shan and Yakal, Emily A. and Yang, Tansy and Zhao, Arthur and George, Reed and Jain, Viren and Jayaraman, Vivek and Korff, Wyatt and Meissner, Geoffrey W. and Romani, Sandro and Funke, Jan and Knecht, Christopher and Saalfeld, Stephan and Scheffer, Louis K. and Waddell, Scott and Card, Gwyneth M. and Ribeiro, Carlos and Reiser, Michael B. and Hess, Harald F. and Rubin, Gerald M. and Jefferis, Gregory S.X.E.}},
  journal   = {{Cell}},
  volume    = {{189}},
  number    = {{18}},
  pages     = {{5504-5526.e15}},
  year      = {{2026}},
  month     = {{9}},
  publisher = {{Elsevier}},
  doi       = {{10.1016/j.cell.2026.08.015}},
  url       = {{https://doi.org/10.1016/j.cell.2026.08.015}},
  note      = {{Preprint: bioRxiv 10.1101/2025.10.09.680999. Data: MaleCNS v1.0, CC-BY 4.0, https://male-cns.janelia.org}}
}}
```

## License

Released under **CC-BY 4.0** by the FlyEM Project Team (HHMI Janelia), the University of Cambridge (Dept. of Zoology),
the MRC Laboratory of Molecular Biology, and Google Research. This repository is a repackaging and carries the same
license; attribution to the original authors (above) is required.

Official site: https://male-cns.janelia.org · neuPrint dataset `male-cns:v1.0`.

This repository is the data backbone of [FlyGPT](https://github.com/QuixiAI/FlyGPT), which trains a language model
whose recurrent architecture is a subgraph of this connectome, with its own trainable edge weights.
"""


if __name__ == "__main__":
    main()
