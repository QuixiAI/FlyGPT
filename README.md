# FlyGPT

## Train a Fruit-Fly Connectome to Write Shakespeare

### Project goal

Build a real language model whose recurrent core is derived from the fruit-fly connectome, train it from scratch on Tiny Shakespeare, and produce a demo people instantly understand.

The central question is:

> **Can a fruit-fly connectome learn to write Shakespeare?**

The project is optimized for:

* fast iteration;
* visible results;
* technically honest claims;
* compelling generated text;
* a strong visual demo;
* easy comparison against a scrambled control;
* a launch story that fits in one post or short video.

The primary success condition is **not publication-quality biological fidelity**.

The primary success condition is:

> We can type `ROMEO:` into a model built around a real fruit-fly wiring graph and get recognizably Shakespeare-like text back.

---

# 1. Viral claim hierarchy

The strongest claim we can honestly earn is:

> **I trained a fruit-fly brain to write Shakespeare.**

More precise supporting line:

> The model uses the wiring topology of a real fruit-fly nervous system as its recurrent neural architecture.

Even stronger, if the control works:

> **The real fly wiring learned language better than the same network after I scrambled its brain.**

That is the ideal result.

---

# 2. What we are actually building

Pipeline:

```text
character
   ↓
learned input embedding
   ↓
selected fly neurons
   ↓
trainable recurrent fly-connectome graph
   ↓
selected output neurons
   ↓
linear readout
   ↓
next-character prediction
```

The graph determines **which neurons may communicate**.

Training determines **the strength of those connections**.

We are not trying to simulate the full biophysics of a living fly.

We are using the connectome as a sparse recurrent neural-network architecture.

---

# 3. Dataset

Use Tiny Shakespeare.

Character-level language modeling.

Vocabulary:

```text
~65 characters
```

Task:

```text
given previous characters
predict next character
```

Example:

```text
ROMEO:
But soft, what light...
```

The advantage of Tiny Shakespeare is that:

* it is tiny;
* training is fast;
* character-level modeling avoids tokenizer complexity;
* generated output is easy to judge visually;
* Karpathy has made it culturally recognizable in AI circles.

---

# 4. First architecture

Do not begin with the entire fly nervous system.

Start with a real connected subgraph.

### V0 target

```text
5,000–10,000 neurons
```

with enough recurrent connectivity to be interesting.

Each biological neuron becomes one scalar recurrent unit.

For neuron `i`:

```text
h_i = scalar hidden state
```

Each real directed neuron-to-neuron connection becomes one trainable weight.

Update:

```text
new_state_i =
tanh(
    recurrent_input_i
    + character_input_i
    + bias_i
)
```

Optionally include a simple residual/leak:

```text
h_new =
(1 - leak) * h_old
+ leak * tanh(...)
```

Keep it simple until it works.

---

# 5. Biological fidelity rules

For V0, preserve:

* actual neuron nodes;
* actual directed connectivity;
* actual graph sparsity.

Do not initially require:

* biological neurotransmitter signs;
* spiking neurons;
* synaptic delays;
* membrane potentials;
* neuromodulators;
* exact synaptic strength;
* biologically accurate learning rules.

Those are later experiments.

The key claim is:

> **The architecture comes from the fly connectome.**

Not:

> **This is a perfect simulation of a fly brain.**

---

# 6. Input interface

Characters must enter the graph somehow.

Use:

```text
65-character vocabulary
        ↓
small learned embedding
        ↓
projection into K input neurons
```

V0:

```text
embedding_dim = 32 or 64
K_input = 256–1024
```

The input adapter should remain tiny compared with the recurrent graph.

For the first version, choose input neurons using a deterministic rule.

Possible options:

```text
random fixed neurons
high-influence nodes
annotated sensory neurons
```

The fastest option wins initially.

Later compare biological sensory nodes against random interface nodes.

---

# 7. Output interface

Select:

```text
512–4096 output neurons
```

Read their hidden states.

Then:

```text
selected states
      ↓
linear layer
      ↓
65 logits
```

Softmax gives the next-character distribution.

Again, keep the readout small.

---

# 8. Temporal model

The fly graph is recurrent.

For every character:

```text
inject character
      ↓
run graph update
      ↓
predict next character
      ↓
retain hidden state
```

Initially:

```text
1 graph update per character
```

If information propagation is too weak:

```text
2
4
```

microsteps per character.

Do not add complexity unless needed.

---

# 9. Training objective

Standard next-character cross-entropy.

No Qwen.

No distillation.

No pretrained embeddings.

No language model teacher.

That makes the result much cleaner:

> **The fly learned Shakespeare directly from Shakespeare.**

---

# 10. Training stack

Use PyTorch.

Recommended starting setup:

```text
optimizer: AdamW
precision: BF16 if available
gradient clipping: 1.0
context length: 64
batch size: whatever fits
learning rate: ~3e-4 recurrent
learning rate: ~1e-3 adapters
```

Use truncated backpropagation through time.

Start with:

```text
sequence length = 64
```

Then move toward:

```text
128
256
```

only after training is stable.

---

# 11. Implementation priority

The only thing that matters initially is:

> **Can the graph learn?**

Do not spend days optimizing kernels before proving that.

First implementation can use:

* edge list;
* source indices;
* destination indices;
* trainable edge weights;
* scatter-add message passing.

Conceptually:

```python
messages = state[src] * edge_weight
incoming = scatter_add(messages, dst)
new_state = tanh(incoming + input)
```

Batch dimension comes later if necessary.

Correctness first.

Speed second.

---

# 12. Build order

## Step 1 — Tiny synthetic graph

Before touching the fly:

```text
100–500 units
```

Train it on:

* repeated sequences;
* delayed-copy tasks;
* tiny text.

Goal:

Confirm recurrent training and gradients work.

Time spent here should be minimal.

---

## Step 2 — Tiny fly graph

Take:

```text
~1,000 real fly neurons
```

Train on a tiny Shakespeare excerpt.

Goal:

**Overfit it.**

Success looks like:

```text
training loss → very low
```

and generated text reproduces the training pattern.

If this fails, debug before scaling.

---

## Step 3 — FlyGPT V0

Scale to:

```text
5k–10k real fly neurons
```

Dataset:

```text
Tiny Shakespeare
```

Train from scratch.

Goal:

Validation loss falls and generated text becomes recognizably language-like.

This is the first potentially shareable milestone.

---

# 13. First viral checkpoint

As soon as output starts resembling:

```text
KING RICHARD:
My lord, I shall not...
```

save the checkpoint.

Immediately build a generation script:

```bash
python generate.py --prompt "ROMEO:"
```

Output:

```text
ROMEO:
...
```

Do not wait for perfect training.

The first coherent fly-generated sentence is content.

---

# 14. Scrambled-fly control

Once RealFly works, create:

## ScrambledFly

Same:

* number of neurons;
* number of edges;
* edge parameter count;
* input/output interfaces;
* training code;
* dataset;
* optimizer.

But randomly rewire the graph.

Then train it identically.

This gives the simplest viral comparison:

```text
REAL FLY
vs
SCRAMBLED FLY
```

No need to start with sophisticated degree-preserving controls.

Those can come later.

If RealFly performs noticeably better:

**that becomes the story.**

---

# 15. Scoreboard

Keep launch metrics simple.

Example:

| Model         | Validation loss | Generated text           |
| ------------- | --------------: | ------------------------ |
| Real Fly      |            1.72 | readable-ish Shakespeare |
| Scrambled Fly |            2.01 | mostly nonsense          |
| Tiny RNN      |            1.65 | readable                 |
| Tiny GPT      |            1.48 | better                   |

Numbers are placeholders.

The most important visible comparison is:

```text
same training
same graph size
real wiring vs scrambled wiring
```

---

# 16. Full-connectome version

Only after the smaller version clearly works.

Scale toward:

```text
~166k neurons
```

and the full MaleCNS graph.

This becomes:

> **Full FlyGPT**

Potential headline:

> **I trained an entire digitally reconstructed fruit-fly nervous system to predict Shakespeare.**

But this is V2.

Do not block V0 on full scale.

---

# 17. Visualization

The demo should make the project visually obvious.

Screen layout:

```text
┌──────────────────────────────────────┐
│ FlyGPT                               │
├──────────────┬──────────────┬────────┤
│ Prompt       │ Fly brain    │ Output │
│              │ activity     │        │
│ ROMEO:       │  •••••••     │ ROMEO: │
│              │ •••••••••    │ ...    │
└──────────────┴──────────────┴────────┘
```

Do not attempt to render 166k neurons live.

Visualize:

* sampled neurons;
* coarse anatomical regions;
* active nodes;
* current propagation;
* maybe top recurrent paths.

The visualization is explanatory, not scientific instrumentation.

---

# 18. Demo interaction

Ideal public demo:

User enters:

```text
ROMEO:
```

or:

```text
KING:
```

or arbitrary text.

Then presses:

```text
GENERATE
```

The fly-network visualization animates.

Text appears character by character.

Optional toggle:

```text
[ Real Fly ] [ Scrambled Fly ]
```

This is extremely useful for demos and video.

---

# 19. Launch video

Target:

```text
30–60 seconds
```

Script:

### Hook

> “Google and Janelia mapped the wiring of an entire fruit-fly nervous system.”

### Cut

Show connectome visualization.

> “So I turned it into a language model.”

### Cut

Show architecture.

```text
Shakespeare
    ↓
fruit fly
    ↓
next character
```

### Cut

Show loss descending.

### Cut

Type:

```text
ROMEO:
```

### Reveal

Fly generates Shakespeare-ish text.

### Final comparison

```text
REAL FLY
vs
SCRAMBLED FLY
```

### Closing line

If RealFly wins:

> “Apparently scrambling its brain makes it worse at Shakespeare.”

That is the clip.

---

# 20. Twitter/X launch thread

Opening post:

> **I trained a fruit-fly brain to write Shakespeare.**
>
> Scientists recently mapped the wiring of an entire fruit-fly nervous system.
>
> I used that connectome as the architecture of a neural network and trained it from scratch on Tiny Shakespeare.
>
> Here's what happened:

Attach video.

Second post:

> This isn't a Transformer pretending to be a fly.
>
> Each node corresponds to a real fly neuron, and recurrent connections are constrained by the actual connectome.
>
> Gradient descent learns the connection strengths.

Third:

```text
Real fly:      loss X
Scrambled fly: loss Y
```

Fourth:

Show favorite generated passage.

Fifth:

Link code/demo.

---

# 21. Honesty constraints

Do not say:

> “We simulated a biological fruit fly brain.”

Unless we eventually actually model the biological dynamics.

Prefer:

> “We trained a neural network using the fruit-fly connectome as its architecture.”

Do not say:

> “The fly understands Shakespeare.”

Say:

> “The network learned character-level Shakespeare statistics.”

The joke can be stronger than the technical claim.

That combination works well.

---

# 22. Failure is still content

If FlyGPT completely fails:

That itself can become:

> **I tried to teach a fruit fly Shakespeare. It went badly.**

Then show:

```text
ROMEO:
fd;kkH::?aa...
```

and investigate why.

Possible follow-up:

> “How many neurons does Shakespeare require?”

That can become a scaling experiment:

```text
1k fly
5k fly
10k fly
50k fly
full fly
```

Plot language quality against neuron count.

That is still interesting and very shareable.

---

# 23. High-value experiments

After the first working demo:

### Experiment A

Real fly vs scrambled fly.

### Experiment B

How many fly neurons are needed before recognizable language appears?

```text
1k
5k
10k
25k
50k
100k
full
```

### Experiment C

Which brain regions matter?

Remove sections and retrain/evaluate.

### Experiment D

Can FlyGPT learn something other than Shakespeare?

Examples:

```text
Python source
Linux kernel text
Twitter posts
Bible
Dr. Seuss-like public-domain text
```

Prefer datasets with clear legal/public-domain status.

### Experiment E

Can the fly learn word-level/token-level language?

Only later.

---

# 24. Optimization target

The first objective is not:

```text
best possible validation loss
```

It is:

```text
first recognizably coherent generated text
```

After that:

```text
best RealFly vs ScrambledFly difference
```

After that:

```text
scale
```

---

# 25. Viral milestone ladder

## Milestone 1

**A real fly subgraph overfits text.**

Internal milestone.

## Milestone 2

**FlyGPT produces recognizable Shakespeare.**

Post-worthy.

## Milestone 3

**Real fly beats scrambled fly.**

Very post-worthy.

## Milestone 4

**Interactive visualization.**

Launch-worthy.

## Milestone 5

**Full connectome works.**

Major launch.

## Milestone 6

**Interesting emergent difference between biological and randomized topology.**

Potential research story on top of the viral project.

---

# 26. Repository

```text
flygpt/
├── README.md
├── pyproject.toml
├── train.py                 # python train.py configs/fly_5k.yaml
├── generate.py              # python generate.py --ckpt checkpoints/fly_5k.pt --prompt "ROMEO:"
├── evaluate.py              # val loss per checkpoint -> results/scoreboard.md
├── visualize.py             # static neuron-activity plot for a prompt
│
├── flygpt/                  # the library
│   ├── graph.py             # load edges, pick a connected subgraph, choose I/O neurons
│   ├── scramble.py          # ScrambledFly: same nodes/edges/params, random wiring
│   ├── model.py             # FlyRNN: embed -> input neurons -> scatter-add recurrence -> readout
│   ├── data.py              # Tiny Shakespeare + toy tasks, char vocab, TBPTT batches
│   ├── baselines.py         # tiny RNN and tiny GPT for the scoreboard
│   ├── config.py            # yaml -> dataclasses
│   └── checkpoint.py
│
├── configs/
│   ├── synthetic_300.yaml   # step 1: random graph, delayed-copy task
│   ├── fly_1k.yaml          # step 2: overfit an excerpt
│   ├── fly_5k.yaml          # step 3: FlyGPT V0
│   ├── fly_10k.yaml
│   ├── fly_full.yaml        # V2
│   ├── scrambled_5k.yaml    # fly_5k + scramble: true, nothing else
│   ├── baseline_rnn.yaml
│   └── baseline_gpt.yaml
│
├── data/
│   ├── shakespeare/download.sh
│   └── fly/                 # fetch_malecns.py -> build_edges.py -> edges.parquet
│
├── results/                 # scoreboard.md + samples/ are committed; they are the content
├── checkpoints/             # gitignored
├── runs/                    # gitignored
├── experiments/             # section 23, added once V0 talks
├── demo/                    # milestone 4
└── tests/
```

Getting started:

```bash
uv venv && uv pip install -e ".[dev]"
pytest
python train.py configs/synthetic_300.yaml          # step 1, ~10 s on CPU
export NEUPRINT_TOKEN=...                           # neuprint.janelia.org -> Account
python data/fly/fetch_malecns.py && python data/fly/build_edges.py
python train.py configs/fly_1k.yaml                 # step 2
python train.py configs/fly_5k.yaml                 # step 3
python train.py configs/scrambled_5k.yaml           # the control
python evaluate.py checkpoints/*.pt
```
---

# 27. README opening

```text
# FlyGPT 🪰

Can a fruit-fly brain learn to write Shakespeare?

Researchers mapped the complete wiring diagram of a fruit-fly
central nervous system.

FlyGPT uses that biological connectome as the recurrent architecture
of a language model.

We train its connection weights from scratch on Tiny Shakespeare.

No pretrained language model.
No transformer hidden inside.
Just Shakespeare → fly → next character.
```

Then immediately show generated output.

---

# 28. Immediate build target

Build this first:

```text
FlyGPT V0

neurons:        5,000–10,000
graph:          real MaleCNS subgraph
hidden/neuron:  1 scalar
task:           character-level language modeling
dataset:        Tiny Shakespeare
context:        64
microsteps:     1
optimizer:      AdamW
input neurons:  256–512
output neurons: 512–1024
```

Required result:

```text
validation loss visibly decreases
```

Then:

```text
generate("ROMEO:")
```

The moment output becomes recognizably Shakespeare-like, stop optimizing long enough to capture it.

---

# 29. The one experiment that matters most

After the first working model:

```text
REAL CONNECTOME
       vs
SAME NETWORK WITH EDGES SCRAMBLED
```

Train both from random initialization.

Same everything else.

If the real fly wins, that is the result to lead with.

If it doesn't, lead with the fact that the fly learns at all.

---

# 30. Final project definition

> **FlyGPT is a language model trained from scratch on Shakespeare whose recurrent architecture is constrained by the wiring diagram of a real fruit-fly nervous system.**

The build philosophy:

> **Make the fly talk first. Do neuroscience afterward.**

