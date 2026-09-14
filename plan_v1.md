# FlyGPT v1 — Spec (draft, pending approval to freeze)

**Status: draft.** v0's discipline was to freeze the spec and then execute (`plan.md` §21). This document is
the proposal; it becomes frozen on the user's approval, and after that changes are recorded in
`notes/decisions.md`, never as new revisions. Open design questions are listed in §14 and should be settled
before freezing.

---

## 0. Objective

v0 asked whether the fly's wiring learns Shakespeare better than the same neurons with degree-preserving
scrambled connections. The answer was **no detectable difference at 5,000 neurons** (`conclusions.md` §2).

v1 asks the question v0 was structurally unable to answer:

> **Does the fly's modular organization let one nervous system learn several tasks without them destroying
> each other, the way a real nervous system does?**

The viral framing stays honest and gets better:

> I taught a fruit-fly brain Shakespeare. Then I taught the same brain a second thing, and asked whether it
> forgot the first.

### Why this is the right next question

v0 destroyed 76.7% regional segregation down to 30.2%, turning 1,765 direct optic-lobe-to-central-brain edges
into 1,455,916, and validation loss moved by 0.004 nats (`conclusions.md` §4). That is not evidence the
structure is useless. Next-character prediction read out from high-degree hub neurons never asks the network
to keep two things separate, so segregation has no value to it. v1 gives the network a reason to care.

### What must remain true

The core principle from v0 carries over verbatim:

> **Don't accidentally cripple the fly and then call its failure a result, and don't accidentally favor the fly
> and then call its win a result.**

v1 introduces one genuinely new risk: biological input/output placement. §5 and §6 address it by construction.

---

## 1. What v0 established (inputs to this design)

| fact | number | source |
|---|--:|---|
| fly learns Shakespeare | 1.578 nats/char vs 2.482 bigram | `results/scoreboard.md` |
| wiring effect, single task, 5 paired seeds | +0.004 nats, verdict "no detectable difference" | `results/claim_degree_preserving*.json` |
| trainable edges beat frozen reservoir | 0.28 nats | `conclusions.md` §3 |
| dense baselines overfit, fly does not | fly final within 0.02 of best; baselines 0.18–0.53 worse | `results/scoreboard.md` |
| whole connectome trains | 160,514 neurons, 10,402,842 edges, 0.48 s/step on 6 GPUs | `graphs/cns_full/`, run logs |
| real graph is modular | 76.7% within-region vs 30.2% scrambled | `conclusions.md` §4 |

The last two are what make v1 possible: the substrate exists, it trains, and it has the structure under test.

---

## 2. Hypothesis and mechanism

**Hypothesis.** When two tasks enter through anatomically distinct sensory pathways, the real connectome
retains the first task better than a degree-preserving scramble does.

**Mechanism.** In the real fly, traffic between regions passes through small dedicated relay populations:
9,200 visual projection neurons, 1,846 ascending, 1,314 descending. Gradients from a task entering the nerve
cord reach optic-lobe circuitry only through that bottleneck. In the scramble every neuron's degree is
identical, but its partners are global, so the second task's gradients flood the territory the first was
using. The prediction is therefore about **interference**, not about final accuracy on either task alone.

**This is the first hypothesis in the project where the real wiring has a mechanistic reason to win.** It also
predicts the v0 null: a single task has nothing to interfere with.

---

## 3. Substrate: the whole connectome

v1 runs on `cns_full`: every traced neuron in the largest strongly connected component of the
three-synapse-threshold graph. 160,514 neurons, 10,402,842 connections, 102,352,176 synaptic contacts.
Regional composition 91,082 optic lobe, 36,625 central brain, 19,300 nerve cord, 13,454 spanning.

A reduced multi-region subgraph was considered and rejected. The whole nervous system is the object the
hypothesis is about, it already trains, and any reduction is an extra design choice to defend.

Consequence: v1 can finally say "the wiring diagram of the fly central nervous system" rather than "a real
subgraph of" (`plan.md` §19 upgrade condition), provided the run is genuine and reported as one seed until
five exist.

---

## 4. Tasks

Two stages. Stage 1 is the decisive measurement; stage 2 is the biologically faithful version.

### Stage 1 (primary): two character corpora

Both tasks are next-character prediction, so **forgetting is measured in the same units on both** and the
pre-registered threshold in §8 is meaningful.

* **Task A:** Tiny Shakespeare, the v0 corpus, fixed 90/10 split, corpus hash committed.
* **Task B:** a second corpus with clearly different statistics and substantial character overlap. Candidates,
  to be fixed before freezing: Python source, a second natural language in Latin script, or the concatenated
  works of a different author. Requirement: both tasks individually learnable by both conditions to within
  0.05 nats of each other (§9 gate), so interference is not confounded with capacity.
* Shared character vocabulary, union of both corpora. Separate input adapters and readout heads per task; the
  recurrent core is the only shared substrate, so interference can only travel through the wiring.

### Stage 2 (follow-up, gated on stage 1): vision plus language

Image classification through the optic lobe, next-character prediction through the central brain. Faithful to
the anatomy, but the two losses are in different units, which complicates a single interference threshold.
Run only after stage 1, and report descriptively rather than against a pre-registered numeric rule.

---

## 5. Biological input and output placement

This is the central departure from v0, whose §5 chose ports by degree alone.

| | task A | task B |
|---|---|---|
| input population | `ol_sensory`, 3,588 neurons in the graph | `vnc_sensory`, 5,905 neurons |
| readout population | `cb_intrinsic` subset, top 4,096 by in-degree | `descending_neuron`, 1,314 neurons |

Two sensory streams converging on a shared integration center is the fly's actual architecture, and it is what
makes the interference question biological rather than arbitrary.

**Why this stays fair.** v0 chose degree-based ports so that a real-wiring win could not be an artifact of
placing ports where the real graph happens to route well. That concern dissolves here for a specific reason:
**the scramble preserves the neurons and their anatomical labels, and only rewires edges.** So
`ol_sensory` denotes the byte-identical neuron set in every condition. Port placement cannot differ between
conditions because it is defined on nodes, not on connectivity. This must be asserted in code, as v0 asserts
I/O identity today.

**Microsteps.** v0 froze microsteps at 2 so it could never be tuned. v1 cannot inherit that number, because
biological ports are many hops from their readouts where degree-based hub ports were one. Rule: run the §6
path diagnostic on the **real** graph only, before any training, and set microsteps once from the anatomy.
Use that value unchanged for every condition and every seed. This is conservative rather than favorable:
random graphs have shorter paths, so a budget that just suffices for the real graph is more than enough for
the scramble. Record the chosen value and the paths that justified it in `notes/decisions.md`, and never
revisit it in response to a result.

---

## 6. Controls

Three conditions. All share the identical neuron set, per-neuron in-degree and out-degree, edge count, and
synapse-count multiset.

**A. RealFly.** Original connectivity.

**B. DegreePreservedFly.** v0's primary control, unchanged: directed double-edge swaps to a survival plateau,
two independent shuffles that must agree. Destroys regional segregation (measured: 76.7% to 30.2%).

**C. ModularRandomFly.** New, and the reason this experiment is informative rather than merely confirmatory.
Degree-preserving swaps **restricted to preserve the region-to-region edge-count matrix**: accept a swap of
`(a→b, c→d)` to `(a→d, c→b)` only when `region(b) == region(d)`. The result has the fly's modular block
structure with random wiring inside and between blocks.

The three-way comparison separates two conclusions that a two-way comparison would conflate:

| outcome | reading |
|---|---|
| real ≈ modular-random, both better than degree-preserved | modularity matters; the fly's specific wiring does not |
| real better than modular-random | the fly's specific wiring matters beyond its block structure |
| all three equal | the hypothesis fails for this task pair (still a result, §12) |

Every control gets the full v0 diagnostic set plus the regional segregation measurement of
`conclusions.md` §4, logged before training.

---

## 7. Metrics

Sequential training, the sharp test of the hypothesis.

```text
phase 1: train on task A for S steps            -> L_A_after_A   (val loss on A)
phase 2: train on task B for S steps            -> L_A_after_AB, L_B_after_AB
```

* **Forgetting** `F = L_A_after_AB − L_A_after_A`. The primary quantity. Lower is better.
* **Plasticity** `L_B_after_AB`, compared against a task-B-only run of the same length, to confirm B was
  actually learned. A condition that refuses to learn B will trivially not forget A, and must not be scored
  as a win. This is the single most important confound in the design.
* **Paired difference** `Δ_k = F(control_k) − F(real_k)` for seed k, reported directly, alongside every seed
  trace, exactly as v0 reports its differences.
* Interleaved training (both tasks sampled in one stream) is a secondary measurement: it captures interference
  during joint learning rather than catastrophic forgetting, and it is cheaper. Run it after the sequential
  result, not instead of it.
* **Regional learning concentration**, free and immediate: per edge, |value after training − value at init|,
  grouped by region pair. Tells us whether learning localizes without being told to.

---

## 8. Pre-registered claim rule

Fixed before any v1 result is seen, deliberately simple, in the spirit of `plan.md` §12.

* **Paired seeds.** Real, degree-preserved, and modular-random for seed k share task order, data order, batch
  sampling, adapter initialization, and the edge-value RNG stream. 1 seed for a dev look, **5 paired seeds for
  any claim**.
* "The fly's modularity reduces interference" is claimed only if, against DegreePreservedFly, **all 5 paired
  differences have the same sign** and the **mean difference is at least 0.05 nats**, with the plasticity check
  of §7 passing in both conditions.
* "The fly's specific wiring matters beyond its block structure" additionally requires the same two conditions
  against ModularRandomFly.
* Otherwise: **"no detectable difference at this scale"**, reported as a result (§12).
* If measured forgetting is below 0.1 nats in every condition, the task pair is too separable to resolve
  anything and the outcome is reported as **underpowered**, not as a null. This case is pre-registered here
  precisely so it cannot be reinterpreted later.
* `claim.py` is extended to the forgetting metric and prints the only permitted verdict string.

---

## 9. Go/no-go gates

**Before training**

* `cns_full` graph built, all three conditions generated, full diagnostics logged for each.
* Regional segregation measured and logged for all three conditions; ModularRandomFly must match RealFly's
  region-to-region matrix to within 1%, and DegreePreservedFly must not.
* I/O populations byte-identical across conditions, asserted in code.
* Path diagnostic run on the real graph; microsteps set once from it and recorded.
* Both tasks individually learnable by both conditions to within 0.05 nats of each other.
* Sparse/dense unit test and gradient-reach test pass on the multi-task model.

**Before the 5-seed claim**

* Dev look on 1 seed shows forgetting of at least 0.2 nats in at least one condition. Otherwise the task pair
  is redesigned before spending the full budget.
* Plasticity check passes: every condition learns task B.

**Before any public claim**

* §8 rule passes with 5 paired seeds, and the wording is exactly what `claim.py` prints.

---

## 10. Build order

```text
 1. Batched GPU degree-preserving rewire (prerequisite, see §11) + equality test against the CPU swapper
 2. ModularRandomFly control: region-constrained swaps, with the region-matrix check as a unit test
 3. Multi-task data pipeline: second corpus, shared vocabulary, per-task streams, committed hashes
 4. Per-task adapters and readout heads in FlyRNN; gradient-reach test over both heads
 5. Biological I/O selection by annotation, with the cross-condition identity assertion
 6. Path diagnostic on the real graph; fix microsteps; log it
 7. Interference harness: sequential phases, forgetting and plasticity metrics, claim.py extension
 8. Individual-task learnability gate, both conditions
 9. Dev look: 1 paired seed, 3 conditions
10. If the dev look passes its gate: 5 paired seeds, 3 conditions
11. Regional learning-concentration analysis
12. Interleaved-training secondary measurement
13. Stage 2: vision plus language
```

---

## 11. Cost, measured not guessed

From the v0 whole-CNS run: 0.48 s/step with 6 GPUs in data parallel at global batch 192.

| item | cost |
|---|---|
| one sequential run (two phases, 16,700 steps each) | ~4.5 h on 6 GPUs |
| dev look, 1 seed × 3 conditions | ~14 h |
| full claim, 5 seeds × 3 conditions | ~67 h of 6-GPU time |

Two prerequisites follow from this table.

**The rewire becomes load-bearing.** The degree-preserving swap is a single-threaded Python loop: ~20 minutes
per control seed on 10.4M edges. Five seeds times two control types is over 3 hours of CPU, and
ModularRandomFly needs a new swapper anyway. Build the batched version (propose thousands of swaps per round
on the GPU, apply the non-conflicting ones) as build-order step 1. It produces a different Markov chain from
the committed CPU swapper, so it must be a separate code path, verified to preserve degrees exactly and to
reach the same survival plateau, and the v0 controls must stay reproducible with the old path.

**Stage the spend.** Do not launch 67 hours before the 1-seed dev look has cleared its gate.

---

## 12. Falsification, and why a null is still content

The hypothesis fails if forgetting is the same across conditions. That outcome says the fly's modularity does
not protect learned function under gradient descent, which is a substantive and publishable claim about the
difference between anatomical modularity and functional modularity.

The headline survives either way:

> I gave one fruit-fly brain two things to learn. Here is whether its 1,765-edge bottleneck kept them apart.

---

## 13. What we will and will not say

**Say, if earned:** "The recurrent architecture is the wiring diagram of the fly central nervous system"
(earned only once the whole CNS is genuinely trained and reported with its seed count). · "Two tasks enter
through anatomically distinct sensory populations." · "Compared against the same neurons with
degree-preserving scrambled connections, and against a control that keeps the fly's regional block structure
but randomizes wiring within it, across five paired seeds."

**Do not say:** that the fly "remembers like a brain", that this demonstrates biological memory
consolidation, or anything about a living fly. Do not report a forgetting advantage without the plasticity
check of §7. Do not describe a single-seed dev look as a result.

---

## 14. Open questions to settle before freezing

1. **Task B.** Which second corpus? This is the one choice that most affects whether the experiment can
   resolve anything, because it sets how much the two tasks compete.
2. **Phase length.** 16,700 steps per phase matches the v0 whole-CNS run, but forgetting depends strongly on
   how long phase 2 runs. Fix one value, or measure forgetting as a curve over phase-2 steps (better, and only
   marginally more expensive since it reuses one run).
3. **Readout for task A.** Top-in-degree central-brain neurons, or an anatomically named population?
4. **Whether to also run 5k.** A cheap 5k multi-task version would not test the hypothesis, since the
   central-brain core has no optic lobe or nerve cord, but it would debug the harness for hours instead of
   days. Recommended as a harness shakedown only, explicitly not as a result.
5. **Scope of stage 2.** Whether vision is in v1 at all, or is its own v2.
