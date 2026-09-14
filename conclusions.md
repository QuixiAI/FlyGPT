# FlyGPT: conclusions

What we found by training a fruit-fly connectome to write Shakespeare, and what we did not find.
Every number here is produced by the scripts in this repository and lives in `results/`, `graphs/`, or the
run logs. Wording follows the public-claim rules in `plan.md` §19. Status as of 2026-09-14; the sections
marked **provisional** are from runs that are still training.

## The short version

A 5,000-neuron subgraph of the fruit-fly central brain, with one trainable weight per real synaptic
connection, learns to produce Shakespeare-shaped English text. It reaches 1.587 nats per character against a
bigram floor of 2.482, and it lands within 0.04 nats of a size-matched transformer.

Scrambling which neurons connect, while preserving every neuron's in-degree and out-degree, costs almost
nothing on this task: 0.004 nats across five paired seeds. Under our pre-registered rule that is **no
detectable difference at this scale**.

The two results that do separate cleanly are about training and about structure. Training the fly's synapses
beats freezing them by 0.28 nats. And the real nervous system is modular in a way the scramble is not, which
our single-task experiment was designed not to see.

## 1. The fly learns

| | nats/char |
|---|--:|
| unigram baseline on our split | 3.347 |
| bigram baseline on our split | 2.482 |
| FlyGPT, 5k central-brain neurons, best of 5 seeds | 1.578 |
| documented external reference: nanoGPT char model, 10.6M params | ~1.47 |

Text at 1.58 has correct speaker turns, mostly real English words, lines that scan, and no coherence across
lines. Generation from `QuixiAI/FlyGPT`, prompt in the first line:

```
Juliet:  Romeo, what vexes you this hour?

MERCUTIO:
You may migor change I none and to humb lies?

KING RICHARD III:
She, we mis well be to the made for hear;
Am come him thy been since are be a pime of him.

CLARENCE:
By the bosom of his joy here is your part.
```

The model never answers as Romeo. It treats the prompt as a script line and continues with the next speaker,
which is what a character model trained on play text does.

## 2. The wiring question: no detectable difference at 5,000 neurons

The pre-registered rule (`plan.md` §12, fixed before any result was seen): claim that wiring matters only if
all five paired differences have the same sign **and** the mean is at least 0.05 nats.

Paired difference is scrambled minus real, per seed, everything else held identical.

| run length | per-seed Δ | mean Δ | signs | verdict |
|---|---|--:|---|---|
| 20k steps | +0.018, +0.013, +0.010, +0.008, +0.007 | +0.011 | 5/5 positive | no detectable difference at this scale |
| 100k steps, final | +0.008, +0.011, +0.003, −0.002, −0.004 | +0.003 | mixed | no detectable difference at this scale |
| 100k steps, best | +0.004, +0.006, +0.001, +0.009, +0.001 | +0.004 | 5/5 positive | no detectable difference at this scale |

The 20k column is the more interesting one to explain. At 20k steps the real wiring led on every seed, and it
led at every checkpoint from step 1,000 onward. With five times the training the scramble catches up. So the
real connectome's advantage on this task is a **learning-speed effect, not a final-accuracy effect**, and it
disappears when both are trained to convergence.

This is a negative result on the headline question, and it is a real one: same neurons, same edge count, same
in-degree and out-degree per neuron, same data order, same adapter initialization, same edge-value RNG stream,
five paired seeds, and the gap is 0.2% of the loss.

## 3. Against engineered baselines, the fly trades accuracy for robustness

All at roughly 578k parameters, three seeds each, identical data, split, context of 64, and training loop.

| model | params | best val | reached at | final val | ms/step |
|---|--:|--:|--:|--:|--:|
| GRU | 577,545 | 1.522 | early | 2.048 | 4 |
| tiny transformer, 4 layers | 564,993 | 1.548 | 21k–28k | 1.724 | 5 |
| dense tanh RNN | 578,055 | 1.581 | ~7k | 2.013 | 4 |
| **FlyGPT, real wiring** | 578,197 | **1.587** | 76k–94k | **1.609** | 18 |
| FlyGPT, scrambled wiring | 578,197 | 1.591 | 76k–94k | 1.613 | 18 |
| FlyGPT, frozen edges (reservoir) | 43,873 trainable | 1.854 | 76k–94k | 1.869 | 18 |

Three things to read from this table.

**The transformer wins, narrowly.** 1.548 versus 1.587 is about 4% in per-character perplexity, and it gets
there in a quarter of the steps at a third of the cost per step. For scale, a 19x larger transformer
(nanoGPT, 10.6M params) reaches only ~1.47 on this corpus, so Tiny Shakespeare is close to saturated at this
size and all these differences are compressed.

**Every dense baseline collapses; the fly does not.** GRU, RNN, and transformer all overfit hard after their
best step, ending 0.18 to 0.53 nats worse. The fly ends within 0.02 of its best. Its fixed sparse wiring
limits what it can memorize. With standard best-checkpoint selection that buys little; under a fixed step
budget it matters.

**Training the connectome is what makes it competitive.** The same graph used as a frozen reservoir, training
only the input projection and readout, reaches 1.854. Training the 524,324 edge values is worth 0.28 nats.
That gap is the entire distance between this project and the frozen-reservoir approach, and it is seven times
larger than the gap to the transformer.

## 4. The structure the experiment could not see

We built the whole traced nervous system: 160,514 neurons and 10,402,842 connections in the largest strongly
connected component at a three-synapse threshold. Then we measured how the degree-preserving scramble changes
its regional organization.

| | real fly | scrambled |
|---|--:|--:|
| edges that stay inside one region | 76.7% | 30.2% |
| direct optic lobe to central brain edges | 1,765 | 1,455,916 |
| optic lobe to nerve cord edges | 0 | random |
| nerve cord to central brain edges | 1 | random |

The fly is modular with extremely narrow interfaces. Nearly all traffic between regions passes through a
small dedicated relay population: 9,200 visual projection neurons, 1,846 ascending, 1,314 descending. The
degree-preserving scramble destroys this completely while preserving every degree exactly, turning 1,765
optic-to-brain edges into 1.46 million.

We destroyed that organization and the loss moved by 0.004 nats. The honest conclusion is not that the
structure is useless. It is that **next-character prediction, read out from high-degree hub neurons, never
asks the network to keep two things separate**, so segregation has no value to it. The interface was chosen by
degree alone (`plan.md` §5) specifically so that a real-wiring win could not be an artifact of port placement.
That choice also made the experiment blind to modularity.

## 5. Scale: non-monotonic, and unresolved

| graph | neurons | edges | best val | basis |
|---|--:|--:|--:|---|
| cb5k, central-brain core | 5,000 | 524,324 | 1.587 (real), 1.591 (scrambled) | 5 paired seeds, complete |
| cb10k, central-brain core | 10,000 | 1,039,667 | 1.561 (scrambled only) | 1 seed, complete; real half killed at 43% |
| cns_full, whole nervous system | 160,514 | 10,402,842 | 1.596 (real) | 1 seed, complete |

On the one seed and condition where 5k and 10k can be compared directly, degree-preserving at seed 1, doubling
the neuron count helped: 1.5816 at 5k against 1.5613 at 10k. The whole nervous system, at eighteen times the
5k neuron count, came out at 1.596, slightly worse than the 5k real mean of 1.587.

So the honest statement is that **the ladder is non-monotonic and we did not finish measuring it.** Going from
5k to 10k improved the model; going to the whole connectome did not. The 10k real run was stopped at 43%, so
no paired difference exists at that rung, and every entry past 5k is a single seed.

What is solid is the overfitting evidence. The whole nervous system ended at train loss 1.333 against
validation 1.596 after about 200 epochs of a 1.1-million-character corpus. A model that fits the training text
that much better than held-out text is not capacity-limited, which is why more neurons stopped paying at the
top of the ladder. Reaching coherent multi-line text plausibly needs a longer context and more data, but that
is a hypothesis this project did not test.

Generation quality reads better at 160k neurons than at 5k, with correctly spelled rare characters from across
the canon and more grammatical lines, which is a reminder that validation loss at this resolution is a coarse
instrument.

## 6. What we would test next

The multi-task hypothesis follows directly from §4 and is the first version of the wiring question where the
real graph has a mechanistic reason to win. Train task A, then task B, and measure how much A degrades. The
prediction is that the real wiring forgets less, because the scramble lets task B's gradients flood the
territory task A was using while the real graph confines them behind a 1,765-edge bottleneck. It needs
biological input and output placement (sensory in, descending out) instead of degree-based ports, which stays
fair: scrambling preserves the neurons and their anatomical labels, so "optic lobe input neurons" denotes the
identical set in both conditions.

It is specified in [`plan_v1.md`](plan_v1.md), on the whole connectome, with a third control that keeps the
fly's regional block structure but randomizes wiring inside it.

Also open, from `plan.md` §10 and §14: longer context, synapse-count initialization, a neurotransmitter sign
constraint, a microstep sweep, and finishing the scaling ladder.

## 7. What we will and will not say

Supported by these results:

- The recurrent architecture is a real subgraph of the fruit-fly brain connectome.
- We train the connection strengths with gradient descent.
- This is not a biological simulation of a living fly.
- At 5,000 neurons, the fly's specific wiring does not measurably beat the same neurons with degree-preserving
  scrambled connections, across five paired seeds.
- Training the connectome's synapses beats using the same wiring as a fixed reservoir by 0.28 nats.
- The fly graph performs within a few percent of a size-matched transformer and resists overfitting where
  dense baselines do not.

Not supported, and not to be said:

- That the fly beats the scrambled fly. The gap is within seed spread.
- That the fly beats a transformer. It does not, at matched parameters.
- Anything about a living fly understanding, learning, or writing English.

## Reproducing these numbers

```bash
python evaluate.py configs/launch_100k.yaml          # the scoreboard in §3
python claim.py configs/launch_100k.yaml --seeds 1 2 3 4 5 [--metric best_val]   # §2
python plot.py configs/launch_100k.yaml              # loss curves
```

Artifacts: connectome at [QuixiAI/MaleCNS](https://huggingface.co/QuixiAI/MaleCNS), trained models at
[QuixiAI/FlyGPT](https://huggingface.co/QuixiAI/FlyGPT), kernels at
[QuixiAI/connectome-kernels](https://github.com/QuixiAI/connectome-kernels). Per-decision history in
`notes/decisions.md`; operational state in `handoff.md`.
