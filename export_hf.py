#!/usr/bin/env python
"""Export a FlyGPT graph (untrained "init") or a trained checkpoint as a Hugging Face model repo.

    python export_hf.py --config configs/launch.yaml --condition real --seed 1 --out ~/flygpt-hf        # init
    python export_hf.py --ckpt checkpoints/flygpt-v0/cb5k/real_seed1.pt --out ~/flygpt-hf-shakespeare  # trained

Repo layout (loads with AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)):
    model.safetensors          graph.* int32/int64 anatomy, recurrent.* + adapters in bf16
    config.json                FlyGPTConfig + auto_map
    configuration_flygpt.py, modeling_flygpt.py
    tokenizer.json, tokenizer_config.json      character-level tokenizer (65 symbols)
    graph_metadata.json        extraction stats, diagnostics, provenance, hash
    README.md                  model card (CC-BY 4.0 attribution, citation, claim wording)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

from flygpt import Config, build_model, load_condition
from flygpt.checkpoint import load_checkpoint
from flygpt.data import CharVocab

HF_DIR = Path(__file__).parent / "hf"


def build_char_tokenizer(chars: list[str]) -> Tokenizer:
    tok = Tokenizer(models.WordLevel({c: i for i, c in enumerate(chars)}, unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok.decoder = decoders.Fuse()
    return tok


def export(model, cfg: Config, vocab: CharVocab, graph, condition: str, seed: int, out: Path,
           training_state: dict, name: str):
    out.mkdir(parents=True, exist_ok=True)
    sd = model.state_dict()
    N, E = model.n, sd["edge_values"].numel()
    dst, src = sd["indices"][0], sd["indices"][1]
    bf = lambda t: t.detach().to(torch.bfloat16).contiguous()
    tensors = {
        "graph.edge_index": torch.stack([src, dst]).to(torch.int32).contiguous(),
        "graph.synapse_count": torch.as_tensor(graph.weight).to(torch.int32).contiguous(),
        "graph.node_id": torch.as_tensor(graph.ids).to(torch.int64).contiguous(),
        "graph.input_nodes": sd["input_nodes"].to(torch.int64).contiguous(),
        "graph.output_nodes": sd["output_nodes"].to(torch.int64).contiguous(),
        "recurrent.edge_values": bf(sd["edge_values"]),
        "recurrent.bias": bf(sd["bias"]),
        "recurrent.raw_leak": bf(sd["raw_leak"]),
        "embed.weight": bf(sd["embed.weight"]),
        "input_proj.weight": bf(sd["inp.weight"]),
        "input_proj.bias": bf(sd["inp.bias"]),
        "lm_head.weight": bf(sd["readout.weight"]),
        "lm_head.bias": bf(sd["readout.bias"]),
    }
    assert tensors["graph.edge_index"].max() < N and (tensors["graph.edge_index"].long() == torch.stack([src, dst])).all()
    save_file(tensors, out / "model.safetensors",
              metadata={"format": "pt", "source": "MaleCNS v1.0 (CC-BY 4.0)", "architecture": "FlyGPTForCausalLM"})

    gdir = cfg.graph_dir
    stats = json.loads((gdir / "subgraph_stats.json").read_text()) if (gdir / "subgraph_stats.json").exists() else {}
    diag_path = Path(load_condition(cfg, condition, seed)[3]["dir"]) / "diagnostics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    graph_meta = {"graph_name": cfg.graph_name, "condition": condition, "control_seed": seed, "source": cfg.graph.source,
                  "region_filter": cfg.graph.region_filter, "min_synapses": cfg.graph.min_synapses,
                  "selector": cfg.graph.selector, "target_neurons": cfg.graph.target_neurons,
                  "hash": (gdir / "subgraph_hash.txt").read_text().strip() if (gdir / "subgraph_hash.txt").exists() else None,
                  "stats": stats, "diagnostics": diag}
    (out / "graph_metadata.json").write_text(json.dumps(graph_meta, indent=2))

    config = {
        "model_type": "flygpt", "architectures": ["FlyGPTForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_flygpt.FlyGPTConfig", "AutoModelForCausalLM": "modeling_flygpt.FlyGPTForCausalLM"},
        "vocab_size": len(vocab), "num_neurons": N, "num_edges": E, "embedding_dim": cfg.model.embed_dim,
        "num_input_neurons": int(sd["input_nodes"].numel()), "num_output_neurons": int(sd["output_nodes"].numel()),
        "microsteps": cfg.sequence.microsteps, "activation": cfg.model.activation, "learned_leak": cfg.model.learned_leak,
        "leak_init": cfg.model.leak_init, "degree_normalization": cfg.model.degree_normalization,
        "init_scale": cfg.model.init_scale, "dtype": "bfloat16",
        "graph": {k: graph_meta[k] for k in ("graph_name", "condition", "control_seed", "source", "region_filter", "min_synapses", "hash")},
        "training_state": training_state,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "generation_config.json").write_text(json.dumps({"do_sample": True, "temperature": 0.8, "max_new_tokens": 300}, indent=2))
    for f in ("configuration_flygpt.py", "modeling_flygpt.py"):
        shutil.copy(HF_DIR / f, out / f)

    build_char_tokenizer(vocab.chars).save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 10_000_000,
        "clean_up_tokenization_spaces": False, "add_prefix_space": False}, indent=2))
    (out / "README.md").write_text(model_card(name, config, graph_meta, training_state, tensors))
    sizes = {k: f"{tuple(v.shape)} {str(v.dtype).replace('torch.', '')}" for k, v in tensors.items()}
    print(json.dumps({"out": str(out), "num_neurons": N, "num_edges": E, "tensors": sizes,
                      "safetensors_mb": round((out / "model.safetensors").stat().st_size / 1e6, 2)}, indent=2))


def result_section() -> str:
    """The five-seed paired result from results/claim_degree_preserving.json, worded exactly as claim.py prints it."""
    p = Path("results/claim_degree_preserving.json")
    if not p.exists():
        return ""
    d = json.loads(p.read_text()); rows = d["paired"]["rows"]; v = d["verdict"]
    table = "\n".join(f"| {r['seed']} | {r['real']:.4f} | {r['control']:.4f} | {r['delta']:+.4f} |" for r in rows)
    return f"""## Result: does the wiring matter?

Pre-registered rule (before any result was seen): "wiring matters" is claimed only if all {v['n_required']} paired
differences Δ = loss(scrambled) − loss(real) have the same sign **and** the mean Δ is at least {v['min_mean_diff_nats']} nats/char.

| seed | real | degree-preserving scramble | Δ |
|--:|--:|--:|--:|
{table}
| **mean** | **{d['paired']['mean_real']:.4f}** | **{d['paired']['mean_control']:.4f}** | **{d['paired']['mean_delta']:+.4f}** |

Validation loss in nats/char at the end of 20,000 steps, same data order, batches, adapter init and edge-value RNG
stream per seed. Bigram reference on this split: 2.482. All five differences favour the real wiring, but the mean
gap is below the pre-registered minimum effect, so the verdict is: **{v['verdict']}.** The fly connectome learns
Shakespeare; whether its specific wiring helps, beyond its degree sequence, is not resolved at 5,000 neurons.

"""


def model_card(name: str, config: dict, gm: dict, ts: dict, tensors: dict) -> str:
    result_section_text = result_section() if ts.get("status") == "trained" and gm.get("graph_name") == "cb5k" else ""
    st = gm.get("stats", {})
    d = gm.get("diagnostics", {}).get("diagnostics", {})
    prov = gm.get("diagnostics", {}).get("provenance", {})
    rows = "\n".join(f"| `{k}` | {tuple(v.shape)} | {str(v.dtype).replace('torch.', '')} | {v.numel() * v.element_size() / 1e6:.2f} MB |"
                     for k, v in tensors.items())
    cond = config["graph"]["condition"]
    wiring = {"real": "the original MaleCNS wiring",
              "degree_preserving": f"a degree-preserving scramble of the MaleCNS wiring (control seed {config['graph']['control_seed']}; "
                                   f"edge survival {prov.get('edge_survival', float('nan')):.3f})",
              "uniform_random": "a uniform-random rewiring (same N and E)"}.get(cond, cond)
    status = ("**Untrained initialization.** The recurrent edge values and adapters are the exact random initialization "
              f"that FlyGPT's `train.py` starts from for seed {ts.get('seed')}. Text generated from this checkpoint is noise by design."
              if ts.get("status") == "init" else
              f"**Trained.** Condition `{ts.get('condition')}`, seed {ts.get('seed')}, step {ts.get('step')}, "
              f"validation loss {ts.get('val_loss'):.4f} nats/char on the fixed Tiny Shakespeare split.")
    return f"""---
license: cc-by-4.0
language: [en]
library_name: transformers
pipeline_tag: text-generation
base_model: QuixiAI/MaleCNS
base_model_relation: finetune
datasets: [karpathy/tiny_shakespeare]
tags: [connectome, fruit-fly, drosophila, malecns, recurrent, sparse, tiny-shakespeare, custom_code]
---

# {name}

A character-level language model whose recurrent architecture is **a real subgraph of the fruit-fly brain
connectome** ([MaleCNS v1.0](https://male-cns.janelia.org/)). Unlike the earlier frozen-reservoir approach in
[ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf), which keeps the connectome's synaptic weights fixed and
trains only the projections and readout, FlyGPT **trains one value per real synaptic connection with gradient
descent** while keeping the fly's edge topology fixed, and compares the result against the same neurons with
degree-preserving scrambled connections across paired seeds.

**Base model: [QuixiAI/MaleCNS](https://huggingface.co/QuixiAI/MaleCNS)**, the lossless packaging of the MaleCNS v1.0
connectivity tables. FlyGPT's graph is extracted from it deterministically (`build_graph.py`, revision pinned in
`data/fly/build_edges.py`); `graph.node_id` and `graph.synapse_count` map every edge back to that repository.

This checkpoint's wiring is {wiring}.

{status}

This is not a biological simulation of a living fly. The "weights" in the MaleCNS release are anatomical synapse
counts; they are stored here as `graph.synapse_count` and are **not** the model's parameters.

## The graph

Every number below is produced by FlyGPT's extraction script (`build_graph.py`), not typed by hand.

| | |
|---|---|
| Source | MaleCNS v1.0 flat connectome (`gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`) |
| Candidate pool | central brain: `superclass` starting with `cb_` ({st.get('candidate_pool', '?'):,} neurons) |
| Minimum synapses per connection | {config['graph']['min_synapses']} (engineering choice, not a biological claim) |
| Extraction | largest SCC → largest directed (k,k)-core with ≥ target nodes (k = {st.get('core_k', '?')}) → trim by weighted degree |
| Neurons used | {config['num_neurons']:,} |
| Directed connections used | {config['num_edges']:,} |
| Synaptic contacts represented | {int(st.get('synaptic_contacts', 0)):,} |
| Largest SCC fraction | {d.get('largest_scc_fraction', '?')} |
| Reciprocal pairs | {d.get('reciprocal_pairs', '?'):,} |
| Input / output neurons | top {config['num_input_neurons']} by out-degree / top {config['num_output_neurons']} by in-degree |
| Input→output shortest path (median / p90 / max hops) | {d.get('io_path_median', '?')} / {d.get('io_path_p90', '?')} / {d.get('io_path_max', '?')} |
| Graph hash | `{config['graph']['hash']}` |

`graph.node_id` holds the MaleCNS body ids, so every neuron maps back to the release.

## What is in `model.safetensors`

| tensor | shape | dtype | size |
|---|---|---|---|
{rows}

`graph.*` is the anatomy (integer, never trained). `recurrent.*`, `embed.*`, `input_proj.*`, `lm_head.*` are the
learned state, stored in bf16. The sparse recurrent matmul is rebuilt in fp32 at runtime (rows = destination,
columns = source), with each incoming edge scaled by `1/sqrt(in_degree)`.

## Dynamics

```text
character → embedding ({config['embedding_dim']}) → linear → {config['num_input_neurons']} input neurons
proposal_i = tanh( Σ_j W_ij h_j / sqrt(in_degree_i) + external_input_i + bias_i )
h_i ← (1 − leak_i) h_i + leak_i · proposal_i        ({config['microsteps']} microsteps per character, leak_i = sigmoid(raw_leak_i))
{config['num_output_neurons']} output neuron states → linear → {config['vocab_size']} logits
```

## Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("QuixiAI/FlyGPT")
model = AutoModelForCausalLM.from_pretrained("QuixiAI/FlyGPT", trust_remote_code=True, dtype=torch.float32)

ids = tok("ROMEO:", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=300, do_sample=True, temperature=0.8)
print(tok.decode(out[0]))

# The degree-preserving scrambled control (same neurons, same degrees, shuffled wiring), for comparison:
scrambled = AutoModelForCausalLM.from_pretrained("QuixiAI/FlyGPT", subfolder="scrambled", trust_remote_code=True, dtype=torch.float32)
print(tok.decode(scrambled.generate(ids, max_new_tokens=300, do_sample=True, temperature=0.8)[0]))

# Neuron activity, for visualization: [1, T, 5000] states after each character, plus MaleCNS body ids
with torch.no_grad():
    states = model(ids).state                                   # [B, N] after the last character
body_ids = model.graph.node_id                                  # index -> MaleCNS body id, for lookup in QuixiAI/MaleCNS
```

The tokenizer is strict: only the 65 characters of Tiny Shakespeare are encodable. `generate()` carries the neuron
state between characters instead of a KV cache.

## Training

The recurrent core has one trainable weight per real synaptic connection. With the
[connectome-kernels](https://github.com/QuixiAI/connectome-kernels) package installed, the model's forward pass
runs on fused CUDA kernels (about 13× faster than `torch.sparse`, identical gradients); without it, it falls back
to `torch.sparse` automatically.

```python
# Fine-tune / continue training FlyGPT on Tiny Shakespeare (character-level).
# pip install transformers safetensors
# pip install --no-build-isolation git+https://github.com/QuixiAI/connectome-kernels   # fused CUDA path, ~13x faster
import requests, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("QuixiAI/FlyGPT")
model = AutoModelForCausalLM.from_pretrained("QuixiAI/FlyGPT", trust_remote_code=True, dtype=torch.float32).cuda()
# start from the untrained initialization instead:  subfolder="init"

text = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").text
data = torch.tensor(tok(text).input_ids)
train, val = data[: int(0.9 * len(data))], data[int(0.9 * len(data)):]   # FlyGPT's fixed 90/10 split

def batch(split, B=32, T=64):
    i = torch.randint(0, len(split) - T - 1, (B,))
    x = torch.stack([split[j : j + T] for j in i]); y = torch.stack([split[j + 1 : j + T + 1] for j in i])
    return x.cuda(), y.cuda()

recurrent = list(model.recurrent.parameters())                         # one weight per real synapse, bias, leak
adapters = [p for n, p in model.named_parameters() if not n.startswith("recurrent.")]
opt = torch.optim.AdamW([{{"params": adapters, "lr": 1e-3}}, {{"params": recurrent, "lr": 3e-4}}], weight_decay=0.01)

for step in range(1, 501):
    x, y = batch(train)
    logits = model(x).logits                                           # [B, T, 65]; state resets to zero per window
    loss = F.cross_entropy(logits.reshape(-1, 65), y.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 100 == 0:
        with torch.no_grad():
            vx, vy = batch(val); vl = F.cross_entropy(model(vx).logits.reshape(-1, 65), vy.reshape(-1))
        print(f"step {{step}}  train {{loss.item():.3f}}  val {{vl.item():.3f}}")

model.save_pretrained("flygpt-finetuned"); tok.save_pretrained("flygpt-finetuned")
```

{result_section_text}## Citation

If you use this model, please cite it, its base model, and the MaleCNS dataset paper.

This model:

```bibtex
@misc{{hartford2026flygpt,
  title        = {{FlyGPT: a language model whose recurrent architecture is a real subgraph of the fruit-fly connectome}},
  author       = {{Hartford, Eric}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/QuixiAI/FlyGPT}}}},
  note         = {{Base model: QuixiAI/MaleCNS (MaleCNS v1.0, Berg et al. 2026, CC-BY 4.0). Code: https://github.com/QuixiAI/FlyGPT}}
}}
```

The base model (lossless connectome packaging):

```bibtex
@misc{{hartford2026malecns,
  title        = {{QuixiAI/MaleCNS: the MaleCNS v1.0 fruit-fly connectome as lossless Safetensors}},
  author       = {{Hartford, Eric}},
  year         = {{2026}},
  publisher    = {{Hugging Face}},
  doi          = {{10.57967/hf/10410}},
  howpublished = {{\\url{{https://huggingface.co/QuixiAI/MaleCNS}}}},
  note         = {{Repackaging of Berg et al. (2026), CC-BY 4.0}}
}}
```

The dataset (required by the CC-BY 4.0 license):

```bibtex
@article{{berg2026malecns,
  title     = {{Sexual dimorphism in the complete {{Drosophila}} male central nervous system connectome}},
  author    = {{Berg, Stuart and Beckett, Isabella R. and Costa, Marta and Schlegel, Philipp and Januszewski, Michał and Marin, Elizabeth C. and Nern, Aljoscha and Preibisch, Stephan and Qiu, Wei and Takemura, Shin-ya and Fragniere, Alexandra M.C. and Champion, Andrew S. and Adjavon, Diane-Yayra and Cook, Michael and Gkantia, Marina and Hayworth, Kenneth J. and Huang, Gary B. and Katz, William T. and Kämpf, Florian and Lu, Zhiyuan and Ordish, Christopher and Paterson, Tyler and Stürner, Tomke and Trautman, Eric T. and Whittle, Catherine R. and Burnett, Laura E. and Hoeller, Judith and Li, Feng and Loesche, Frank and Morris, Billy J. and Pietzsch, Tobias and Pleijzier, Markus W. and Silva, Valeria and Yin, Yijie and Ali, Iris and Badalamente, Griffin and Bates, Alexander Shakeel and Beresford, Rory J. and Bogovic, John and Brooks, Paul and Cachero, Sebastian and Canino, Brandon S. and Chaisrisawatsuk, Bhumpanya and Clements, Jody and Crowe, Arthur and de Haan Vicente, Inês and Dempsey, Georgia and Donà, Erika and Dos Santos, Márcia and Dreher, Marisa and Dunne, Christopher R. and Eichler, Katharina and Finley-May, Samantha and Flynn, Miriam A. and Hameed, Imran and Hopkins, Gary Patrick and Hubbard, Philip M. and Kiassat, Ladann and Kovalyak, Julie and Lauchie, Shirley A. and Leonard, Meghan and Lohff, Alanna and Longden, Kit D. and Maldonado, Charli A. and Moitra, Ilina and Moon, Sung Soo and Mooney, Caroline and Munnelly, Eva J. and Okeoma, Nneoma and Olbris, Donald J. and Pai, Anika and Patel, Birava and Phillips, Emily M. and Plaza, Stephen M. and Richards, Alana and Rivas Salinas, Jennifer and Roberts, Ruairí J.V. and Rogers, Edward M. and Scott, Ashley L. and Scuderi, Louis A. and Seenivasan, Pavithraa and Serratosa Capdevila, Laia and Smith, Claire and Svirskas, Rob and Takemura, Satoko and Tastekin, Ibrahim and Thomson, Alexander and Umayam, Lowell and Walsh, John J. and Whittome, Holly and Xu, C. Shan and Yakal, Emily A. and Yang, Tansy and Zhao, Arthur and George, Reed and Jain, Viren and Jayaraman, Vivek and Korff, Wyatt and Meissner, Geoffrey W. and Romani, Sandro and Funke, Jan and Knecht, Christopher and Saalfeld, Stephan and Scheffer, Louis K. and Waddell, Scott and Card, Gwyneth M. and Ribeiro, Carlos and Reiser, Michael B. and Hess, Harald F. and Rubin, Gerald M. and Jefferis, Gregory S.X.E.}},
  journal   = {{Cell}},
  volume    = {{189}},
  number    = {{18}},
  pages     = {{5504--5526.e15}},
  year      = {{2026}},
  month     = sep,
  publisher = {{Elsevier}},
  doi       = {{10.1016/j.cell.2026.08.015}},
  url       = {{https://doi.org/10.1016/j.cell.2026.08.015}},
  note      = {{Preprint: bioRxiv 10.1101/2025.10.09.680999. Data: MaleCNS v1.0, CC-BY 4.0, https://male-cns.janelia.org}}
}}
```

## License

The connectome is released under CC-BY 4.0 by the FlyEM Project Team (HHMI Janelia), the University of Cambridge,
the MRC Laboratory of Molecular Biology, and Google Research. This checkpoint is a derivative and carries the same
license.

Prior art: [ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf) (frozen MaleCNS reservoir LM) and
[eob/gpt-fly](https://huggingface.co/eob/gpt-fly) (FlyWire-masked GPT-2). Code and experiment:
[github.com/QuixiAI/FlyGPT](https://github.com/QuixiAI/FlyGPT).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/launch.yaml")
    ap.add_argument("--condition", default="real")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ckpt", help="trained checkpoint (.pt); overrides --config/--condition/--seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", help="repo name for the model card, e.g. QuixiAI/FlyGPT-MaleCNS-5K-Init")
    args = ap.parse_args()

    if args.ckpt:
        model, meta, vocab, cfg, ck = load_checkpoint(args.ckpt, "cpu")
        condition, seed = ck["condition"], ck["seed"]
        ts = {"status": "trained", "condition": condition, "seed": seed, "step": ck["step"], "val_loss": ck["val_loss"],
              "checkpoint": str(args.ckpt), "project": cfg.project}
    else:
        cfg = Config.load(args.config)
        split = json.loads(Path("data/shakespeare/split.json").read_text())
        vocab = CharVocab(split["vocab"])
        condition, seed = args.condition, args.seed
        model, meta = build_model(cfg, condition, seed, len(vocab))
        ts = {"status": "init", "condition": condition, "seed": seed, "project": cfg.project}
    if condition == "frozen":
        sys.exit("export the frozen reservoir as condition=real with its checkpoint; 'frozen' shares the real graph")
    graph = load_condition(cfg, condition, seed)[0]
    name = args.name or f"FlyGPT-{cfg.graph_name}-{condition}-{ts['status']}"
    export(model.cpu(), cfg, vocab, graph, condition, seed, Path(args.out).expanduser(), ts, name)


if __name__ == "__main__":
    main()
