# Repository layout

Maps README spec v4 onto files. The README is the spec and stays frozen; this file is the map.

```text
flygpt/
├── README.md                    # spec v4 (frozen)
├── pyproject.toml
├── configs/
│   ├── launch.yaml              # §13 — the one launch config, no grid
│   └── dev_1k.yaml              # §15 "1k overfits an excerpt" gate; differs only in size/corpus/length
│
├── prepare_data.py              # §2  download, hash, fix 90/10 split, measure unigram/bigram  -> data/shakespeare/split.json
├── build_graph.py               # §3-6 extract dense core, build every control × seed, degree I/O, diagnostics, path gate
├── train.py                     # §9  one condition × one seed; logs everything §9 lists; fixed-prompt generations
├── claim.py                     # §12 paired differences + pre-registered claim rule
├── evaluate.py                  # §11 scoreboard: model, params, val loss mean ± range over seeds
├── plot.py                      # §9  launch chart from logs: val loss vs step / wall-clock / chars
├── generate.py                  # sample from a checkpoint
├── visualize.py                 # §17 static activity raster (sampled neurons, explanatory only)
├── export_hf.py                 # FlyGPT graph/checkpoint -> Hugging Face repo (safetensors bf16 + trust_remote_code)
├── scripts/launch_cb5k.sh       # detached per-GPU queues for real + degree_preserving × seeds 1–5
├── hf/                          # self-contained HF model code copied into exported repos
│   ├── configuration_flygpt.py, modeling_flygpt.py     # FlyGPTForCausalLM (state carried instead of a KV cache)
│   └── configuration_malecns.py, modeling_malecns.py   # MaleCNSConnectome: lossless connectome tensors + subset masks
│
├── flygpt/
│   ├── config.py                # §13 schema as dataclasses
│   ├── data.py                  # corpus, split, hash, reference losses, seeded TBPTT batches
│   ├── connectome/
│   │   ├── malecns.py           # §1  MaleCNS v1.0 tables + region filter (§3.1)
│   │   ├── extract.py           # §3.3 directed-core procedure, artifacts, stats
│   │   ├── controls.py          # §4  degree-preserving swaps w/ plateau check, uniform random, §4.2 weights
│   │   ├── interface.py         # §5  I/O by degree only
│   │   ├── diagnostics.py       # §4.1/§6 SCC, reciprocity, I→O path lengths, gate
│   │   └── synthetic.py         # fake connectome for tests and smoke runs
│   ├── model.py                 # §7/§8 FlyRNN: sparse.mm rows=dst, learned leak, degree norm, dense reference; FrozenFly
│   ├── baselines.py             # §11 tanh RNN / GRU / tiny Transformer, parameter matching
│   ├── analysis.py              # §12 Δ_k, sign test, min effect
│   └── checkpoint.py
│
├── data/
│   ├── shakespeare/split.json   # committed: corpus sha256, split boundary, vocab, unigram/bigram nats
│   └── fly/                     # fetch_malecns.py -> build_edges.py -> edges.parquet + neurons.parquet (gitignored)
│                                #   export_malecns_hf.py -> ~/malecns-hf (full traced connectome as a HF repo)
│
├── graphs/<graph_name>/         # build_graph.py output. Committed: ids, config, hash, stats, diagnostics, gate.json.
│   ├── subgraph_node_ids.txt    #   Gitignored: *.pt edge tensors (rebuildable, hash-checked).
│   ├── subgraph_config.yaml
│   ├── subgraph_hash.txt
│   ├── subgraph_stats.json
│   ├── real/                    # edges.pt, input_nodes.txt, output_nodes.txt, diagnostics.json
│   ├── degree_preserving_seed<k>/
│   └── gate.json
│
├── runs/<project>/<graph>/<condition>_seed<k>/   # gitignored: log.jsonl, generations.jsonl, meta.json
├── checkpoints/<project>/<graph>/<condition>_seed<k>.pt
├── results/                     # scoreboard.md, claim_*.json, plots — generated, committed
├── notes/decisions.md           # §21 engineering decisions after the freeze; informal expectations live here only
├── demo/                        # §17, gated on a trained model
└── tests/                       # sparse==dense, gradient reach, degree preservation, I/O identity, gate, claim rule, e2e smoke
```

## Conditions

`--condition` on train.py selects the graph and model:

| condition | graph | what trains |
|---|---|---|
| `real` | real subgraph | edges, bias, leak, adapters |
| `degree_preserving` | rewired, seed k | same |
| `uniform_random` | rewired, seed k | same (follow-up only) |
| `frozen` | real subgraph | adapters only (§11 reservoir gate) |
| `rnn` / `gru` / `transformer` | none | parameter-matched baseline |

Paired seed k: same data order, same adapter init, same edge-value RNG stream across conditions.

## Order of operations

```bash
uv venv && uv pip install -e ".[dev]"
pytest                                                  # sparse/dense, gradient reach, controls, gate, claim rule, e2e

python prepare_data.py                                  # split.json + reference losses
python data/fly/fetch_malecns.py                        # official bulk files -> data/fly/raw/ (md5-verified; --neuprint for the API)
python data/fly/build_edges.py --region-column superclass   # check the printed superclass -> region crosstab
python build_graph.py configs/dev_1k.yaml               # must print "all gates passed"
python train.py configs/dev_1k.yaml --condition frozen  # reservoir gate: must beat bigram
python train.py configs/dev_1k.yaml --condition real    # 1k overfit gate

python build_graph.py configs/launch.yaml
for k in 1 2 3; do
  python train.py configs/launch.yaml --condition real --seed $k &
  python train.py configs/launch.yaml --condition degree_preserving --seed $k &
done; wait
python claim.py configs/launch.yaml --seeds 1 2 3       # dev look; the claim needs 5
python evaluate.py configs/launch.yaml && python plot.py configs/launch.yaml
```

Smoke everything without MaleCNS: `python build_graph.py configs/launch.yaml --synthetic`.
