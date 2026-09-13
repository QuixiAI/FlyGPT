"""The Hugging Face export must reproduce the FlyRNN forward pass (up to bf16 rounding of the stored values)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def synthetic_graph(tmp_path_factory):
    """Build a small synthetic graph with build_graph.py --synthetic and a config pointing at it."""
    import yaml
    d = tmp_path_factory.mktemp("hfexport")
    cfg = yaml.safe_load((ROOT / "configs/launch.yaml").read_text())
    cfg.update(project="hf-test", graph_name="hf_synth")
    cfg["graph"]["target_neurons"] = 300
    cfg["interface"].update(input_nodes=16, output_nodes=32)
    cfg["evaluation"]["seeds_claim"] = [1]
    p = d / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, "build_graph.py", str(p), "--synthetic"], cwd=ROOT, check=True, capture_output=True)
    return p, d


def test_export_matches_flyrnn(synthetic_graph):
    cfg_path, d = synthetic_graph
    out = d / "repo"
    subprocess.run([sys.executable, "export_hf.py", "--config", str(cfg_path), "--condition", "real", "--seed", "3",
                    "--out", str(out)], cwd=ROOT, check=True, capture_output=True)
    for f in ["model.safetensors", "config.json", "modeling_flygpt.py", "configuration_flygpt.py", "tokenizer.json",
              "tokenizer_config.json", "graph_metadata.json", "README.md"]:
        assert (out / f).exists(), f

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors import safe_open
    from flygpt import Config, build_model

    with safe_open(out / "model.safetensors", "pt") as f:
        dt = {k: f.get_slice(k).get_dtype() for k in f.keys()}
    assert dt["graph.edge_index"] == "I32" and dt["graph.synapse_count"] == "I32" and dt["graph.node_id"] == "I64"
    assert all(dt[k] == "BF16" for k in dt if not k.startswith("graph."))

    hf = AutoModelForCausalLM.from_pretrained(out, trust_remote_code=True, dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(out)
    text = "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    ids = tok(text, return_tensors="pt").input_ids
    assert tok.decode(ids[0]) == text  # exact round trip incl. newline and spaces

    cfg = Config.load(cfg_path)
    ref, _ = build_model(cfg, "real", 3, len(tok))
    with torch.no_grad():  # round the reference to the same bf16 values the repo stores
        for p in ref.parameters():
            p.copy_(p.to(torch.bfloat16).float())
        ref_logits, ref_state = ref(ids)
        o = hf(ids)
    assert torch.allclose(o.logits, ref_logits, atol=1e-4), (o.logits - ref_logits).abs().max()
    assert torch.allclose(o.state, ref_state, atol=1e-5)

    # the graph round-trips: same edges, same synapse counts, same body ids
    from flygpt import load_condition
    g = load_condition(cfg, "real", 3)[0]
    ei = hf.graph.edge_index.long()
    assert set(zip(ei[0].tolist(), ei[1].tolist())) == set(zip(g.src.tolist(), g.dst.tolist()))
    assert np.array_equal(hf.graph.node_id.numpy(), g.ids)
    assert np.array_equal(hf.graph.synapse_count.numpy(), g.weight.astype(np.int32))

    # generate() carries the neuron state instead of a KV cache
    torch.manual_seed(0)
    gen = hf.generate(ids[:, :8], max_new_tokens=12, do_sample=False)
    assert gen.shape == (1, 20)
    # greedy generation must equal step-by-step argmax with the full-sequence forward
    with torch.no_grad():
        cur = ids[:, :8]
        for _ in range(12):
            cur = torch.cat([cur, hf(cur).logits[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(gen, cur)
