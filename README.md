# FlyGPT 🪰 — Spec v4 (frozen)

## Can a Fruit-Fly Connectome Learn to Write Shakespeare?

---

## 0. Objective

Build a character-level language model trained from scratch on Tiny Shakespeare whose recurrent connectivity is constrained by the wiring diagram of a real fruit-fly nervous system.

The viral question:

> **Can a fruit-fly connectome learn to write Shakespeare?**

The stronger result:

> **Does the real fly wiring learn better than the same neurons, same edges, same degrees, with the connections scrambled?**

The project optimizes for, in order:

1. a working model and genuine generated text quickly;
2. a comparison that survives obvious criticism;
3. a result that is visually understandable;
4. public claims that are technically defensible.

### Core principle

> **Don't accidentally cripple the fly and then call its failure a result — and don't accidentally favor the fly and then call its win a result.**

Both failure modes are addressed by design decisions in this spec, not by post-hoc cleanup.

---

## 1. Data source: MaleCNS v1.0

Use the official MaleCNS v1.0 release (neuron annotations, segment-to-segment connection graph with synapse counts, per-neuron and per-synapse neurotransmitter predictions, neuPrint access).

**Before anything ships:** confirm the data license and required citation, and include both in the README.

**Do not hard-code edge-count or neuron-count claims.** Every number describing the graph is produced by the preprocessing script and logged:

```text
candidate pool (region filter)
neurons used
directed connections used
represented synaptic contacts
minimum synapse threshold
region breakdown of selected neurons
```

### License and citation

The MaleCNS v1.0 dataset is released by the FlyEM Project Team (HHMI Janelia), the University of Cambridge
(Dept. of Zoology), the MRC Laboratory of Molecular Biology, and Google Research under **CC-BY 4.0**
(https://male-cns.janelia.org/download/). FlyGPT reads the official flat-connectome release files from
`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` (md5-verified; see `data/fly/fetch_malecns.py`).
The neuPrint dataset name for the same release is `male-cns:v1.0`.

If you use FlyGPT's graph, cite the dataset paper:

> Berg, S., Beckett, I. R., Costa, M., Schlegel, P., Januszewski, M., Marin, E. C., Nern, A., Preibisch, S., et al. (2026).
> Sexual dimorphism in the complete *Drosophila* male central nervous system connectome. *Cell* 189, 5504–5526.e15.
> https://doi.org/10.1016/j.cell.2026.08.015 (preprint: bioRxiv https://doi.org/10.1101/2025.10.09.680999)

BibTeX:

```bibtex
@article{berg2026malecns,
  title     = {Sexual dimorphism in the complete {Drosophila} male central nervous system connectome},
  author    = {Berg, Stuart and Beckett, Isabella R. and Costa, Marta and Schlegel, Philipp and Januszewski, Michał and Marin, Elizabeth C. and Nern, Aljoscha and Preibisch, Stephan and Qiu, Wei and Takemura, Shin-ya and Fragniere, Alexandra M.C. and Champion, Andrew S. and Adjavon, Diane-Yayra and Cook, Michael and Gkantia, Marina and Hayworth, Kenneth J. and Huang, Gary B. and Katz, William T. and Kämpf, Florian and Lu, Zhiyuan and Ordish, Christopher and Paterson, Tyler and Stürner, Tomke and Trautman, Eric T. and Whittle, Catherine R. and Burnett, Laura E. and Hoeller, Judith and Li, Feng and Loesche, Frank and Morris, Billy J. and Pietzsch, Tobias and Pleijzier, Markus W. and Silva, Valeria and Yin, Yijie and Ali, Iris and Badalamente, Griffin and Bates, Alexander Shakeel and Beresford, Rory J. and Bogovic, John and Brooks, Paul and Cachero, Sebastian and Canino, Brandon S. and Chaisrisawatsuk, Bhumpanya and Clements, Jody and Crowe, Arthur and de Haan Vicente, Inês and Dempsey, Georgia and Donà, Erika and Dos Santos, Márcia and Dreher, Marisa and Dunne, Christopher R. and Eichler, Katharina and Finley-May, Samantha and Flynn, Miriam A. and Hameed, Imran and Hopkins, Gary Patrick and Hubbard, Philip M. and Kiassat, Ladann and Kovalyak, Julie and Lauchie, Shirley A. and Leonard, Meghan and Lohff, Alanna and Longden, Kit D. and Maldonado, Charli A. and Moitra, Ilina and Moon, Sung Soo and Mooney, Caroline and Munnelly, Eva J. and Okeoma, Nneoma and Olbris, Donald J. and Pai, Anika and Patel, Birava and Phillips, Emily M. and Plaza, Stephen M. and Richards, Alana and Rivas Salinas, Jennifer and Roberts, Ruairí J.V. and Rogers, Edward M. and Scott, Ashley L. and Scuderi, Louis A. and Seenivasan, Pavithraa and Serratosa Capdevila, Laia and Smith, Claire and Svirskas, Rob and Takemura, Satoko and Tastekin, Ibrahim and Thomson, Alexander and Umayam, Lowell and Walsh, John J. and Whittome, Holly and Xu, C. Shan and Yakal, Emily A. and Yang, Tansy and Zhao, Arthur and George, Reed and Jain, Viren and Jayaraman, Vivek and Korff, Wyatt and Meissner, Geoffrey W. and Romani, Sandro and Funke, Jan and Knecht, Christopher and Saalfeld, Stephan and Scheffer, Louis K. and Waddell, Scott and Card, Gwyneth M. and Ribeiro, Carlos and Reiser, Michael B. and Hess, Harald F. and Rubin, Gerald M. and Jefferis, Gregory S.X.E.},
  journal   = {Cell},
  volume    = {189},
  number    = {18},
  pages     = {5504-5526.e15},
  year      = {2026},
  month     = {9},
  publisher = {Elsevier},
  doi       = {10.1016/j.cell.2026.08.015},
  url       = {https://doi.org/10.1016/j.cell.2026.08.015},
  note      = {Preprint: bioRxiv 10.1101/2025.10.09.680999. Data: MaleCNS v1.0, CC-BY 4.0, https://male-cns.janelia.org}
}
```

The connectome is consumed through its lossless Hub packaging, https://huggingface.co/QuixiAI/MaleCNS:

```bibtex
@misc{quixiai2026malecns,
  title        = {MaleCNS: the MaleCNS v1.0 fruit-fly connectome as lossless safetensors},
  author       = {Hartford, Eric},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/QuixiAI/MaleCNS}},
  note         = {Repackaging of Berg et al. (2026), CC-BY 4.0}
}
```

The "weights" in the release are anatomical synapse counts, not learned parameters; FlyGPT uses the directed edge set as
the architecture and trains its own value per retained edge.

---

## 2. Dataset

Tiny Shakespeare, character-level, ~65-symbol vocabulary.

Fixed 90/10 train/validation split. Commit the split boundaries and a hash of the corpus.

No pretrained language model, no teacher, no pretrained embeddings.

### Reference losses (nats/char)

Only two kinds of reference numbers appear anywhere public:

* **Measured on our split:** unigram and bigram, computed by the preprocessing script and logged. These are the floor for every gate below.
* **Documented external reference:** nanoGPT's `train_shakespeare_char` config reports a best validation loss of ~1.47 (verify against the repo README at launch time and cite it).

Rough informal expectations for a plain tanh RNN or "readable but funny" text (~1.7–2.0) may live in engineering notes but are never published or put in a table. Anything else in the scoreboard is a baseline we actually ran.

---

## 3. Subgraph extraction

### 3.1 Region filter (decides the story)

MaleCNS includes the ventral nerve cord. A connectivity-driven core extraction can land mostly in the VNC, which turns "fly brain" into "fly spinal cord."

V0 restricts the candidate pool to **central-brain neurons** using the release annotations. Always log the region breakdown of the final selected set. A VNC-inclusive or whole-CNS pool is a later experiment.

### 3.2 Random sampling is prohibited

Randomly picking 5k–10k neurons from ~166k discards almost all induced connectivity and produces a fragmented graph. Its failure would mean nothing.

### 3.3 Deterministic dense-core procedure

```yaml
region_filter: central_brain
target_neurons: 5000        # launch; 10k is scale-up
min_synapses: 3
subgraph_method: directed_core
```

A. **Filter weak edges:** keep `synapse_count >= min_synapses`. Threshold 3 is fixed for launch; `{1, 5}` are follow-ups only if the launch fails. Log it prominently; it is an engineering choice, not a biological claim.

B. **Largest strongly connected component** of the filtered graph — guarantees recurrent paths exist.

C. **Directed-core pruning:** iteratively remove neurons below an in- and out-degree threshold; choose the largest threshold leaving at least `target_neurons`.

D. **Deterministic trim:** if larger than target, rank by `weighted_in_degree + weighted_out_degree`, keep top `target_neurons`, recompute induced graph and SCC. If the SCC drops below target, lower the pruning threshold in C and repeat.

E. **Save exact IDs and hashes:**

```text
subgraph_node_ids.txt
subgraph_edges.pt
subgraph_config.yaml
subgraph_hash.txt
subgraph_stats.json     # N, E, SCC size, degree stats, reciprocity, region breakdown
```

Every control uses exactly these neurons.

---

## 4. Controls (built in the same script as the real graph)

### A. RealFly
Original directed connectivity.

### B. DegreePreservedFly — primary control
Same neurons, same edge count, same in-degree and out-degree per neuron. Randomize via **directed double-edge swaps**:

* reject swaps that create self-loops or duplicate edges;
* **10 × E** accepted swaps is a minimum, not evidence of mixing;
* log the **fraction of original edges that survive** after scrambling. Some survival is unavoidable under degree preservation (hub-to-hub edges are likely in any degree-matched graph), so the test is that the fraction has **plateaued**: run two independent shuffles with different seeds and continue swapping until survival stops decreasing and both shuffles agree;
* fixed RNG seed per control instance; log it.

### C. UniformRandomFly — secondary control
Same N and E, endpoints fully randomized. For visualization and a sanity ordering; not sufficient for the headline claim.

### 4.1 Every control gets the same diagnostics as the real graph

Swaps preserve degrees but **not** the SCC, reachability, path lengths, or reciprocity. For every graph (real and each control) log:

```text
largest SCC size (fraction of N)
reciprocal-pair count
I→O reachable fraction
I→O shortest-path median / p90 / max
```

Expect RealFly to have far more reciprocal pairs than the scrambled graph. That is part of "wiring" and is fair game, but know the number before anyone asks.

### 4.2 Weighted controls
If synapse counts are used for initialization (§10), the DegreePreserved control receives the original multiset of synapse counts randomly reassigned to its shuffled edges. The difference then remains "which neurons connect," not "what magnitudes exist."

---

## 5. Input/output interface — selected by degree only

**Rationale.** If I/O nodes are chosen by reachability on the real graph and reused for the scrambled graph, the ports have been placed where the real wiring happens to route well. A RealFly win would then be partly an artifact of port placement.

**Rule.** Degree is identical across RealFly and DegreePreservedFly, so select I/O sets purely by degree. The resulting node sets are byte-identical across the two conditions.

```text
input nodes  = top 256 by out-degree   (within the subgraph)
output nodes = top 512 by in-degree    (within the subgraph)
```

Overlap between the two sets is allowed; log it. (For UniformRandomFly, degree is not preserved; apply the same rule to its own degrees and note the sets differ.)

### Input path
```text
character → 32–64-dim embedding → linear projection → 256 input neurons
```

### Output path
```text
512 output neuron states → linear → 65 logits → softmax
```

Adapters stay small relative to the recurrent parameter count; log the ratio.

Biological sensory/motor interfaces are a follow-up experiment.

---

## 6. Path-length diagnostic — mandatory, on every graph, before training

For each condition, compute directed shortest paths from the input set to the output set and report:

```text
reachable output fraction
median, p90, max finite shortest path
```

**Microsteps are frozen at 2 for launch.** The path analysis is a go/no-go gate, not a tuner — a model hyperparameter must not depend on a randomly generated control topology.

Gate: on every condition, reachable output fraction is high and the p90 input→output path is compatible with 2 microsteps per character (i.e. information can traverse the graph within a few characters). If any condition fails the gate, fix the subgraph density or the interface design, then re-run the gate. Do not adjust microsteps to rescue a bad graph.

A microstep sweep (1, 4) is a follow-up experiment run identically on all conditions.

Do not launch training against a graph whose outputs are six hops away and then conclude the fly can't learn.

---

## 7. Neuron model

One biological neuron = one scalar hidden state `h_i ∈ ℝ`. No per-neuron vector embeddings.

```text
proposal_i = tanh( normalized_recurrent_input_i + external_input_i + bias_i )
h_i_new    = (1 - leak_i) * h_i + leak_i * proposal_i
```

* Leak is mandatory. `leak_i = sigmoid(raw_leak_i)`, initialized so `leak_i = 0.5`, trainable.
* With `microsteps > 1`, the leak is applied every microstep, so the effective per-character time constant compounds. Log the mean leak and account for this when comparing microstep settings.
* Weights are unconstrained in sign for V0.

### Degree normalization
For edge `j → i`:

```text
effective_weight_ji = learned_weight_ji / sqrt(in_degree_i)
```

Because in-degree is preserved under the primary control, normalization is identical across conditions.

### Stability logging (every run)
```text
mean |h|, fraction |h| > 0.95, activation variance, gradient norm, mean leak
```

---

## 8. Implementation

### 8.1 Batched sparse recurrence from day one

* State: `[B, N]`.
* Recurrent matrix stored **with rows = destination, columns = source** so that `incoming = torch.sparse.mm(W, state.T).T` needs no extra transpose of W. Build the sparse tensor each forward pass from a dense trainable `values` tensor and fixed `indices`.
* COO first for autograd simplicity; benchmark CSR after correctness is established.
* Never materialize `[B, T, E]` messages.
* Keep a slow dense reference implementation for a unit test on a tiny graph; assert equality with the sparse path and assert gradients reach every edge value.

### 8.2 Precision
Sparse matmul support in BF16 is patchy. Keep the sparse recurrent op in **fp32**; use BF16 only for the dense adapters and readout if it is stable.

### 8.3 Compute budget
Do not plan around a FLOP estimate. Sparse autograd memory and throughput routinely diverge from back-of-envelope numbers.

Procedure: benchmark **one** training run at the launch config, log peak memory and tokens/sec, then parallelize conditions and seeds as measured capacity permits.

Expect the dominant risk to be optimization (dead or saturated activations, leak/LR interactions) rather than raw compute, but confirm that with the benchmark rather than assuming it.

---

## 9. Training

```text
optimizer:        AdamW
recurrent LR:     ~3e-4
adapter LR:       ~1e-3
grad clip:        1.0
context:          64  (→128 →256 only if useful)
BPTT:             truncated
initial state:    zeros at sequence boundaries (persistent state = later experiment)
```

### Logged per run
```text
train loss, val loss, step, characters seen, wall-clock, tokens/sec,
GPU memory, grad norm, activation saturation, mean leak
```

Plot val loss against step, wall-clock, and characters seen. The launch chart must regenerate from logs.

### Fixed-prompt generations
Prompts `ROMEO:`, `KING:`, `JULIET:`, `First Citizen:` at steps 0 / 10% / 25% / 50% / 75% / 100%, fixed seed and temperature. This produces the "garbage → punctuation → fragments → speech" narrative.

---

## 10. Later biological variants (gated behind a working model)

* **Synapse-count initialization:** `magnitude ∝ log(1 + synapse_count)`, then degree-normalized. Compare against random magnitudes, using the weighted control from §4.2.
* **Neurotransmitter-sign constraint:** constrain outgoing weight signs by predicted transmitter class where a defensible excitatory/inhibitory mapping exists.
* **Biological I/O:** annotated sensory neurons as inputs, descending/motor-related neurons as outputs.
* **VNC / whole-CNS pool.**

None of these block V0 or the headline.

---

## 11. Baselines

### FrozenFly (reservoir) — first sanity test
Real topology, fixed random-sign weights, degree-normalized, scaled by power-iteration spectral-radius estimate to ~0.9–1.0, frozen. Train only the input adapter and readout.

Gate: must clearly beat the bigram loss (~2.4). If not, inspect topology, spectral behavior, I/O paths, microsteps, leak, normalization before touching the trainable model.

### Parameter-matched engineered baselines (post-launch-candidate)
For each FlyGPT config log recurrent / input / output / total trainable parameters, then build approximately matched:

* vanilla tanh RNN
* GRU (if practical)
* tiny Transformer

Scoreboard columns: model, params, val loss (mean ± range over seeds), notes.

---

## 12. Seeds and the decision rule

* **Paired seeds.** For seed *k*, RealFly and DegreePreservedFly share the same data order, batch sampling, adapter initialization, and edge-value RNG stream (edge sets differ, so recurrent weights cannot be identical, but everything else is). 3 paired seeds for development; **5 paired seeds for any wiring-matters claim**.
* Report the five paired differences `Δ_k = loss(Scrambled_k) − loss(Real_k)` directly, alongside per-condition means and every seed trace. Do not manufacture a pseudo-p-value from min/max ranges.
* **Pre-register the claim rule before seeing results** (kept deliberately simple): "wiring matters" is claimed only if **all 5 paired differences have the same sign** (a 5/5 sign test) *and* the mean difference is at least a stated minimum meaningful effect, set at **0.05 nats** for launch. If the signs are mixed or the effect is smaller, report "no detectable difference at this scale" — still a publishable result (§14).
* For social media, lead with the cleanest representative generation, but publish the paired differences beside it.

---

## 13. Launch configuration — one config, no grid

```yaml
project: flygpt-v0

dataset:
  name: tiny_shakespeare
  split: 0.90

graph:
  source: malecns-v1.0
  region_filter: central_brain
  target_neurons: 5000
  min_synapses: 3
  selector: directed_core
  retain_largest_scc: true

controls:
  - real
  - degree_preserving       # primary
  # uniform_random is a follow-up, not part of the launch comparison

interface:
  input_nodes: 256
  input_rule: top_out_degree
  output_nodes: 512
  output_rule: top_in_degree

model:
  state_dim_per_neuron: 1
  activation: tanh
  learned_leak: true
  leak_init: 0.5
  degree_normalization: true

sequence:
  context: 64
  microsteps: 2                 # frozen; path diagnostic is a gate, not a tuner

training:
  optimizer: adamw
  recurrent_lr: 0.0003
  adapter_lr: 0.001
  grad_clip: 1.0
  sparse_precision: fp32
  adapter_precision: bf16

evaluation:
  seeds_dev: [1, 2, 3]
  seeds_claim: [1, 2, 3, 4, 5]
  paired_seeds: true
  claim_rule: {all_paired_diffs_same_sign: true, min_mean_diff_nats: 0.05}
```

Everything not in this file — threshold sweeps, microstep sweeps, uniform-random control, synapse-count init, NT signs, GRU/Transformer baselines, 10k and full-CNS scale — is gated behind a working launch model.

---

## 14. Failure is still content

If FlyGPT does not learn, or if real and scrambled are indistinguishable, that is a result:

> **I tried to teach a fruit fly Shakespeare. Here's what happened.**

Follow-up experiment: the **neuron-count scaling ladder**.

```text
1k → 5k → 10k → 25k → 50k → 100k → full
```

Plot val loss (real and degree-preserved) against neuron count. "How many fly neurons does Shakespeare require?" is shareable whether or not the wiring matters.

---

## 15. Go/no-go gates

**Before training**
* dense subgraph exists; SCC healthy; region breakdown acceptable
* controls generated; diagnostics logged for every condition
* I/O sets identical across real and degree-preserved
* path-length gate passes on all conditions at microsteps = 2
* scramble edge-survival fraction has plateaued (§4)
* sparse/dense unit test agrees; gradients reach every edge value
* unigram/bigram reference losses computed on the actual split

**Reservoir gate:** FrozenFly beats bigram.

**Before 1k → 5k:** 1k RealFly overfits a small Shakespeare excerpt to near-memorization. If not, the implementation is broken; stop.

**Before 5k → 10k:** declining val loss and structured generation at 5k.

**Before the demo:** at least one reproducibly trained model.

**Before claiming wiring matters:** the §12 paired-seed claim rule passes with 5 seeds.

---

## 16. Build order

```text
 1. Download MaleCNS connectivity; record license/citation
 2. Region filter + deterministic dense-core extractor (+ stats file)
 3. Degree-preserving rewiring (+ uniform-random, cheap)
 4. Degree-based I/O selection
 5. Path-length / SCC / reciprocity diagnostic for every condition
 6. Batched sparse recurrent cell (rows = destination) + dense reference test
 7. Gradient-reach test
 8. FrozenFly reservoir gate
 9. 1k overfit
10. 5k launch config: real + degree-preserved × 3 seeds, run concurrently
11. Generation checkpoints + demo
12. Video
13. 10k scale-up; engineered baselines; biological variants; scaling ladder
14. Full-CNS attempt
```

---

## 17. Demo

```text
┌───────────────────────────────────────────┐
│                FlyGPT 🪰                  │
├───────────┬─────────────────┬─────────────┤
│ Prompt    │ Neural activity │ Completion  │
│ ROMEO:    │     •••••       │ ROMEO: ...  │
├───────────┴─────────────────┴─────────────┤
│ [ REAL FLY ] [ SCRAMBLED FLY ]            │
└───────────────────────────────────────────┘
```

Rendered activity may be sampled or aggregated. Never imply it is a faithful biological simulation.

---

## 18. Launch video (30–60 s)

1. **Hook:** "Scientists mapped the neural wiring of an entire fruit-fly nervous system. So I tried to teach it Shakespeare." Show the connectome.
2. **Architecture:** character → real fly wiring → next character.
3. **Training:** loss falling; generations improving through the fixed-prompt checkpoints.
4. **Demo:** type `ROMEO:`, show completion.
5. **Control:** flip REAL FLY → SCRAMBLED FLY.
6. **Closing line (only if §12 passes):** "Same neurons. Same number of connections. Same in- and out-degree for every neuron. Scramble which neurons connect, and it gets worse."

If §12 does not pass, the closing line is the fact that it learned at all, or the scaling-ladder question.

---

## 19. Public claim rules

**Say (V0):** "The recurrent architecture is a real subgraph of the fruit-fly brain connectome." · "We train the connection strengths with gradient descent." · "This is not a biological simulation of a living fly." · "Compared against the same neurons with degree-preserving scrambled connections, across five paired seeds."

Upgrade "a real subgraph of" to "the wiring diagram of" only when the full central brain (or full CNS) has actually been trained.

**Do not say:** "We uploaded GPT into a fly." · "A living fruit fly learned English." · "The fly understands Shakespeare." · "The fly beats the scrambled fly" if the gap is within seed spread.

Expect the reply "there's no transformer, why GPT?" — the README pre-empts it; lean in.

The joke can be loose. The technical explanation cannot.

---

## 20. Final definition

> **FlyGPT (V0) is a language model trained from scratch on Shakespeare whose recurrent architecture is a real subgraph of the fruit-fly brain connectome — compared, fairly, against the same neurons with their wiring scrambled.**

---

## 21. Spec freeze

This revision is frozen. Further planning has less expected value than finding out whether 1,000 fly neurons can overfit Shakespeare. Changes from here are recorded as decisions in the run logs, not as new spec revisions, until the first trained model exists.

