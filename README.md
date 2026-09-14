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

Findings so far are written up in [`conclusions.md`](conclusions.md). The full specification is [`plan.md`](plan.md) (spec v4, frozen). Engineering decisions made after the freeze are
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

- Data, graph extraction, controls, diagnostics, every pre-training gate: **done on real data.**
- **5k launch, five paired seeds, 100k steps: done.** Validation loss 1.609 nats/char (real) vs 1.613 (degree-preserving
  scramble), bigram reference 2.482. Paired differences +0.008, +0.011, +0.003, −0.002, −0.004. Under the
  pre-registered rule the verdict is **no detectable difference at this scale**: the fly connectome learns
  Shakespeare, and at 5,000 neurons its specific wiring does not measurably beat a degree-matched scramble.
  At 20k steps the real wiring led on all five seeds by 0.011 nats; that edge is a learning-speed effect that
  disappears with full training. Numbers and plots: `results/`, checkpoints on the Hub:
  [QuixiAI/FlyGPT](https://huggingface.co/QuixiAI/FlyGPT).
- **Whole nervous system** (160,514 traced neurons, 10.4M connections, `configs/full_cns.yaml`): graph built, all
  gates pass, training in progress on 6 GPUs with data parallelism.
- **Parameter-matched baselines (RNN, GRU, transformer, 3 seeds each, ~578k parameters): done.** Best validation loss
  GRU 1.522, transformer 1.548, dense RNN 1.581, FlyGPT 1.587. Every dense baseline then overfits badly (final losses
  1.72 to 2.05) while the fly graph stays within 0.02 of its best: at this size the connectome buys no accuracy, but
  its fixed sparse wiring is a strong regularizer. Full table: `results/scoreboard.md`.

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

Code is MIT-licensed (see LICENSE). The connectome is MaleCNS v1.0, released under
**CC-BY 4.0** by the FlyEM Project Team (HHMI Janelia), the University of Cambridge (Dept. of Zoology), the
MRC Laboratory of Molecular Biology, and Google Research. If you use FlyGPT's graph or checkpoints, cite the
connectome packaging and the dataset paper:

```bibtex
@misc{hartford2026malecns,
  title        = {QuixiAI/MaleCNS: the MaleCNS v1.0 fruit-fly connectome as lossless Safetensors},
  author       = {Hartford, Eric},
  year         = {2026},
  publisher    = {Hugging Face},
  doi          = {10.57967/hf/10410},
  howpublished = {\url{https://huggingface.co/QuixiAI/MaleCNS}},
  note         = {Repackaging of Berg et al. (2026), CC-BY 4.0}
}

@article{berg2026malecns,
  title     = {Sexual dimorphism in the complete {Drosophila} male central nervous system connectome},
  author    = {Berg, Stuart and Beckett, Isabella R. and Costa, Marta and Schlegel, Philipp and Januszewski, Michał and Marin, Elizabeth C. and Nern, Aljoscha and Preibisch, Stephan and others},
  journal   = {Cell},
  volume    = {189},
  number    = {18},
  pages     = {5504--5526.e15},
  year      = {2026},
  doi       = {10.1016/j.cell.2026.08.015},
  url       = {https://doi.org/10.1016/j.cell.2026.08.015},
  note      = {Preprint: bioRxiv 10.1101/2025.10.09.680999. Data: MaleCNS v1.0, CC-BY 4.0, https://male-cns.janelia.org}
}
```

The full author list is in [`plan.md`](plan.md) §1.
