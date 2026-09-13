# FlyGPT 🪰

**Can a fruit-fly connectome learn to write Shakespeare?**

FlyGPT is a character-level language model trained from scratch on Tiny Shakespeare whose recurrent
architecture is a real subgraph of the fruit-fly central-brain connectome (MaleCNS v1.0). One biological neuron
is one scalar hidden state; one real synaptic connection is one trainable weight. The fly's wiring fixes *which*
neurons connect; gradient descent learns *how strongly*.

The stronger question, and the one the experiment is built around:

> Does the real fly wiring learn better than the same neurons, same number of connections, same in- and
> out-degree for every neuron, with the connections scrambled?

This is not a biological simulation of a living fly, and there is no transformer in it. The name is a joke; the
comparison is not.

## How it works

```text
character ──► embedding ──► linear ──► 256 input neurons (top out-degree)
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     │  5,000 MaleCNS central-brain neurons        │
                     │  524,324 real directed connections          │
                     │  h ← (1−leak)·h + leak·tanh(W h/√deg + in)  │   × 2 microsteps per character
                     └──────────────────────┬──────────────────────┘
                                            │
              512 output neurons (top in-degree) ──► linear ──► 65 logits
```

- **Graph.** A deterministic dense-core extraction from the central brain (largest strongly connected component,
  then the largest directed k-core of at least 5,000 neurons, then a trim by weighted degree). No random sampling.
  Every number describing the graph is produced by `build_graph.py` and committed under `graphs/`.
- **Control.** The same neurons and edges rewired by directed double-edge swaps until the fraction of surviving
  original edges plateaus (two independent shuffles must agree). In- and out-degree of every neuron are preserved,
  so the input and output neuron sets, chosen by degree only, are byte-identical across conditions.
- **Gates before training.** Path-length and reachability diagnostics on every condition, a sparse-versus-dense
  unit test, a gradient-reach test, a frozen-reservoir sanity run that must beat a bigram model, and a 1k-neuron
  run that must memorize a small excerpt. A graph that fails a gate is fixed at the graph level; the model's
  microsteps are never tuned to rescue it.
- **Decision rule, pre-registered.** Five paired seeds. "Wiring matters" is claimed only if all five paired
  differences have the same sign and the mean difference is at least 0.05 nats/char. Otherwise the result is
  reported as "no detectable difference at this scale", which is also a result.

The full specification is [`plan.md`](plan.md) (spec v4, frozen). Engineering decisions made after the freeze are
logged in [`notes/decisions.md`](notes/decisions.md). The map from spec sections to files is
[`docs/layout.md`](docs/layout.md).

## Data

The connectome is consumed from [QuixiAI/MaleCNS](https://huggingface.co/QuixiAI/MaleCNS), a lossless
safetensors packaging of the MaleCNS v1.0 connectivity tables (every segment, exact synapse counts, release
annotations and neurotransmitter predictions, no modeling choices baked in). It is pinned to a revision in
`data/fly/build_edges.py`, and the graphs built from it are hash-identical to graphs built from the raw release
files. The scripts that build that repository from the official Janelia bucket are in `data/fly/`.

Text is Tiny Shakespeare with a fixed 90/10 split; the corpus hash, split boundary, and measured unigram and
bigram reference losses are committed in `data/shakespeare/split.json`.

## Status

- Data, graph extraction, controls, diagnostics, and every pre-training gate: **done on real data**. The 5k
  launch graph and its five degree-preserving controls pass all gates.
- 5k training (real vs degree-preserving, five paired seeds): **not yet run**.
- Results, plots, and the claim: none yet. Nothing in this README or the model cards claims a result.

## Quickstart

```bash
uv venv && uv pip install -e ".[dev,hf]"
pytest                                            # sparse==dense, gradient reach, controls, gates, claim rule, HF export

python prepare_data.py                            # Tiny Shakespeare split + reference losses
python data/fly/build_edges.py                    # QuixiAI/MaleCNS (pinned) -> data/fly/edges.parquet, neurons.parquet
python build_graph.py configs/dev_1k.yaml         # must print "all gates passed"
python train.py configs/dev_1k.yaml --condition frozen   # reservoir gate: beats bigram
python train.py configs/dev_1k.yaml --condition real     # overfit gate: memorizes the excerpt

python build_graph.py configs/launch.yaml         # graphs/cb5k: real + 5 degree-preserving controls, gate.json
scripts/launch_cb5k.sh                            # all ten paired runs, one per GPU, detached
python evaluate.py configs/launch.yaml            # results/scoreboard.md
python plot.py configs/launch.yaml                # results/val_loss.png, regenerated from logs
python claim.py configs/launch.yaml --seeds 1 2 3 4 5    # the pre-registered rule; wording follows plan.md §19
```

Smoke everything without the connectome: `python build_graph.py configs/launch.yaml --synthetic`.

### Hugging Face export

Trained checkpoints (and the untrained initialization) export to a `trust_remote_code` repo whose safetensors
hold the graph as integers and the learned values in bf16, with `QuixiAI/MaleCNS` declared as the base model:

```bash
python export_hf.py --ckpt checkpoints/flygpt-v0/cb5k/real_seed1.pt --out ~/flygpt-hf --name QuixiAI/FlyGPT
```

## What we will and will not say

Say: "The recurrent architecture is a real subgraph of the fruit-fly brain connectome." · "We train the
connection strengths with gradient descent." · "This is not a biological simulation of a living fly." ·
"Compared against the same neurons with degree-preserving scrambled connections, across five paired seeds."

Do not say: "We uploaded GPT into a fly." · "A living fruit fly learned English." · "The fly understands
Shakespeare." · "The fly beats the scrambled fly" if the gap is within seed spread.

## Prior art

[ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf) uses the MaleCNS central brain as a frozen
echo-state reservoir for a TinyStories language model: the synaptic weights are never trained. FlyGPT's
difference is that the fly's edge values themselves are trained, on a stricter central-brain pool, against a
control that preserves both in- and out-degree, under a pre-registered five-seed rule.
[eob/gpt-fly](https://huggingface.co/eob/gpt-fly) is a GPT-2 whose MLPs are masked with the FlyWire connectome.

## License and citation

The code has no license file yet (to be chosen before release). The connectome is MaleCNS v1.0, released under
**CC-BY 4.0** by the FlyEM Project Team (HHMI Janelia), the University of Cambridge (Dept. of Zoology), the
MRC Laboratory of Molecular Biology, and Google Research. If you use FlyGPT's graph or checkpoints, cite the
connectome packaging and the dataset paper:

```bibtex
@misc{hartford2026malecns,
  title        = {QuixiAI/MaleCNS: the MaleCNS v1.0 fruit-fly connectome as lossless Safetensors},
  author       = {Hartford, Eric},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/QuixiAI/MaleCNS}},
  note         = {Repackaging of Berg et al. (2026), CC-BY 4.0}
}

@article{berg2026malecns,
  title     = {Sexual dimorphism in the complete {Drosophila} male central nervous system connectome},
  author    = {Berg, Stuart and Beckett, Isabella R. and Costa, Marta and Schlegel, Philipp and Januszewski, Michał and Marin, Elizabeth C. and Nern, Aljoscha and Preibisch, Stephan and others},
  journal   = {Cell},
  volume    = {189},
  number    = {18},
  pages     = {5504-5526.e15},
  year      = {2026},
  doi       = {10.1016/j.cell.2026.08.015},
  url       = {https://doi.org/10.1016/j.cell.2026.08.015},
  note      = {Preprint: bioRxiv 10.1101/2025.10.09.680999. Data: MaleCNS v1.0, CC-BY 4.0, https://male-cns.janelia.org}
}
```

The full author list is in [`plan.md`](plan.md) §1.
