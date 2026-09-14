# FlyGPT handoff

Living document for an agent taking over this work. Updated at every milestone; the timestamp below is the
last update. Read it top to bottom before touching anything, then read `plan.md` (the frozen spec),
`notes/decisions.md` (every engineering decision since the freeze, append-only), and `docs/layout.md`.

**Last updated: 2026-09-14 04:05 UTC. All runs stopped at the user's request; every GPU is idle.** Read `conclusions.md` for the findings and `plan_v1.md` for the
proposed next experiment; this file is operational state only. The user stopped active work at this point.

## 1. What this project is, in one paragraph

A character-level language model trained on Tiny Shakespeare whose recurrent wiring is a real subgraph of the
fruit-fly connectome (MaleCNS v1.0), with one trainable weight per real synaptic connection, compared against
the same neurons with degree-preserving scrambled wiring under a pre-registered five-paired-seed rule
(`plan.md` §12). Public claim wording is fixed in `plan.md` §19 and must never be exceeded; `claim.py` prints the
only permitted verdict string.

## 2. Published artifacts (all live)

| what | where | notes |
|---|---|---|
| code | https://github.com/QuixiAI/FlyGPT | MIT. `main` is the only branch. |
| connectome data | https://huggingface.co/QuixiAI/MaleCNS | lossless safetensors of the full MaleCNS table; DOI 10.57967/hf/10410 (bound to card revision 91e441a). Built by `data/fly/export_malecns_hf.py`; local copy `~/malecns-hf`. FlyGPT reads it at the revision pinned in `data/fly/build_edges.py`. |
| kernels | https://github.com/QuixiAI/connectome-kernels | MIT. Fused CUDA sparse recurrence; local checkout `~/connectome-kernels`, installed editable in `~/.venv`. FlyGPT depends on it (`flygpt/kernels.py` delegates). |
| trained models | https://huggingface.co/QuixiAI/FlyGPT | root = cb5k real seed 1 (100k steps, best val 1.5778); `scrambled/` = degree-preserving seed 1; `init/` = untrained init. `base_model: QuixiAI/MaleCNS`. Card has executed inference + training snippets and the five-seed result table. Export: `export_hf.py`. |

Citation keys: `hartford2026malecns` (data), `berg2026malecns` (paper), `hartford2026flygpt` (model),
`hartford2026connectomekernels`. MaleCNS does not cite FlyGPT; FlyGPT cites MaleCNS (deliberate, one-way).

## 3. Results so far (numbers from `results/`, never from memory)

- **cb5k, 5 paired seeds, 20k steps** (`results/cb5k_20k/`): real 1.737 vs scrambled 1.748, all 5 Δ positive,
  mean +0.011 < 0.05 → "no detectable difference at this scale".
- **cb5k, 5 paired seeds, 100k steps** (`results/scoreboard.md`, `results/claim_degree_preserving.json`):
  real 1.609 vs scrambled 1.613 final; best 1.587 vs 1.591; Δ mixed sign, mean +0.003 → same verdict. The 20k
  lead was a learning-speed effect.
- **Baselines** (~578k params, 3 seeds, 100k steps): best val GRU 1.522, transformer 1.548, RNN 1.581, fly 1.587;
  all dense baselines overfit hard afterwards (final 1.72–2.05), the fly does not (final 1.609).
- **FrozenFly reservoir** (adapters only, 43,873 trainable params): best val 1.854 / 1.862 / 1.905 (seed 3 mid-run).
  Trainable edges are worth 0.28 nats over the same wiring frozen.
- **Regional segregation** (whole-CNS graph): 76.7% of real edges stay within one region vs 30.2% scrambled;
  1,765 direct optic-to-central-brain edges vs 1,455,916 scrambled. Measured, in `conclusions.md` §4.
- Reference losses on our split: unigram 3.347, bigram 2.482 (`data/shakespeare/split.json`).

## 4. Nothing is running

All training was stopped at 2026-09-14 04:05 UTC at the user's request. All eight GPUs are idle and no queue
scripts remain. Two runs were killed mid-flight:

| job | where it got to | usable? |
|---|---|---|
| whole CNS degree_preserving seed 1 | step ~1,300 / 16,700 | no; delete `runs/flygpt-v0-fullcns/cns_full/degree_preserving_seed1` and rerun if wanted |
| cb10k real seed 1 | step 42,876 / 100,000, best 1.6345 | partial log and best-val checkpoint exist; not comparable to the completed 100k runs |

Completed and recorded: cb5k at 20k and 100k steps (5 paired seeds each), all three engineered baselines
(3 seeds), FrozenFly (3 seeds), the whole-CNS real run (published to the Hub), and cb10k degree_preserving
seed 1 (best 1.5613).

To resume, the launchers are `scripts/launch_cb5k.sh`, `scripts/launch_baselines.sh`,
`scripts/launch_cns_ddp.sh`, and `scripts/launch_followups.sh`. Each is detached and idempotent apart from
overwriting its own run directory.

## 5. The two loose ends

**The 10k rung of the scaling ladder is half-measured.** `cb10k degree_preserving seed 1` finished at best
1.5613, which is 0.020 *better* than `cb5k degree_preserving seed 1` at 1.5816. The real half was killed at
43%. Finishing it (`CUDA_VISIBLE_DEVICES=0 python train.py configs/scale_10k.yaml --condition real --seed 1`,
about 50 minutes) would give the one paired difference at 10k and settle whether the improvement holds for the
real graph. This is the cheapest remaining piece of real information in the project.

**The whole-CNS wiring comparison does not exist.** Its control was killed almost immediately. A single paired
run at that scale costs about 2.2 hours on 6 GPUs per condition, and by `plan.md` §12 one seed supports no
claim either way.

Neither gap is reflected in any published artifact: `conclusions.md` §5 states plainly that the ladder is
non-monotonic and unfinished, and the Hub card for `full-cns` states that it is one seed with no five-seed
claim.

## 6. How to run things

```bash
cd ~/FlyGPT && source ~/.venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.2 PATH=/usr/local/cuda-13.2/bin:$PATH   # nvcc for connectome-kernels JIT/build
pytest -q                                                                     # ~3 min; 35 tests incl. kernels + HF export

# single run, one GPU
CUDA_VISIBLE_DEVICES=3 python train.py configs/launch_100k.yaml --condition real --seed 1
# data-parallel, N GPUs, one model (global batch = N × batch_size)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node 6 train.py configs/full_cns.yaml --condition real --seed 1
# read-out
python evaluate.py <config>; python plot.py <config>; python claim.py <config> --seeds 1 2 3 4 5 [--metric best_val]
# generate from a checkpoint
python generate.py --ckpt checkpoints/flygpt-v0-100k/cb5k/real_seed1.pt --prompt "ROMEO:" --device cuda:0
```

Quick generation in Python:

```python
import torch; from flygpt.checkpoint import load_checkpoint
m, meta, vocab, cfg, ck = load_checkpoint("checkpoints/flygpt-v0-100k/cb5k/real_seed1.pt", "cuda")
ids = torch.tensor([vocab.encode("ROMEO:")], device="cuda"); torch.manual_seed(0)
print(vocab.decode(m.generate(ids, 300, 0.8)[0].tolist()))
```

Configs: `launch.yaml` (spec §13, 20k steps), `launch_100k.yaml` (same, 100k), `scale_10k.yaml`, `full_cns.yaml`,
`dev_1k.yaml` (gates). Conditions: `real`, `degree_preserving`, `frozen`, `rnn`, `gru`, `transformer`.

## 7. Rules that must not be broken

- `plan.md` is frozen. Never edit it except §1 license/citation. Log decisions in `notes/decisions.md` (append a
  table row; never rewrite rows).
- Every number about a graph comes from `build_graph.py` output (`graphs/<name>/*.json`), never from memory.
- Microsteps stay at 2. The path gate is go/no-go, not a tuner. Do not tune hyperparameters to rescue a result.
- Public wording: only what `claim.py` prints and `plan.md` §19 allows. "No detectable difference at this
  scale" is the current verdict; do not say the fly beats the scramble.
- Paired seeds: real and degree_preserving for seed k share data order, adapter init, edge-value RNG stream.
  Any change to `train.py` data handling must keep that.
- `graphs/**/*.pt`, `runs/`, `checkpoints/`, `data/fly/*.parquet`, `data/fly/raw/` are gitignored and rebuildable.
  Stats, ids, diagnostics, gate.json, results are committed.
- Commit attribution: end commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`
  (or the taking-over agent's line). Git identity is set locally in the repo. `gh` is authenticated; git uses it.
- Hugging Face: `huggingface_hub` token is present for user ehartford (org QuixiAI). MaleCNS card edits: edit the
  template in `data/fly/export_malecns_hf.py` AND `~/malecns-hf/README.md`, then `upload_file` the README.
  FlyGPT card: edit the template in `export_hf.py`, re-export, `upload_folder`.

## 8. Node facts

8× B200 (183 GB), 240 CPUs, Ubuntu 24.04. `~/.venv`: torch 2.14+cu132, transformers 5.17, connectome-kernels
(editable), CUDA toolkit 13.2 at `/usr/local/cuda-13.2`. neuprint.janelia.org is unreachable from this node; data
comes from the Hub repo (or `data/fly/fetch_malecns.py` from the Janelia bucket). Fused kernels: 5k model 18 ms/step,
whole CNS 0.48 s/step on 6 GPUs DDP. The degree-preserving rewire is single-threaded Python: ~20 min per control
seed on the 10.4M-edge graph (a batched GPU version is a known TODO, see decisions).

## 9. Open items / next steps beyond the running jobs

**`plan_v1.md` is the proposed next experiment** (multi-task interference on the whole connectome, three
conditions, pre-registered forgetting rule). It is a draft: §14 lists what the user must settle before it
freezes, and build-order step 1 (batched GPU rewire) is a prerequisite. Do not start spending its 67-hour
budget without the 1-seed dev look clearing its gate.

1. Publish the whole-CNS model (see §5) and write up the scaling ladder 5k → 10k → 160k.
2. Whole-CNS five paired seeds if the wiring question is pursued at that scale (needs the batched rewire first).
3. Follow-ups listed in `plan.md` §10/§14: context 128/256, min_synapses {1,5}, microstep sweep {1,4},
   synapse-count init (§4.2 weighted control exists), neurotransmitter sign constraint, biological I/O.
4. Demo (`plan.md` §17) and video (§18): gated on the trained model, which now exists.
5. Mint a new DOI version for QuixiAI/MaleCNS once its card is final (current DOI is bound to an older revision).
