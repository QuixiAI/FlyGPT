"""build_graph -> train -> claim on a synthetic connectome, in a temp working directory."""
import json, os, subprocess, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_smoke(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs/launch.yaml").read_text())
    cfg.update(project="smoke", graph_name="syn")
    cfg["graph"].update(target_neurons=150, min_synapses=1)
    cfg["interface"].update(input_nodes=16, output_nodes=32)
    cfg["gate"].update(min_reachable_fraction=0.5, max_p90_chars=10)
    cfg["training"].update(steps=6, eval_every=3, batch_size=4, eval_batches=1)
    cfg["evaluation"].update(seeds_claim=[1, 2], seeds_dev=[1], generation_chars=5, generation_fractions=[0.0, 1.0])
    # tiny synthetic corpus so the test needs no network
    corpus = tmp_path / "input.txt"; corpus.write_text("ROMEO: to be or not to be, that is the question.\n" * 40)
    cfg["dataset"].update(path=str(corpus))
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))

    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / a[0]), *a[1:]], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    out = run("build_graph.py", "c.yaml", "--synthetic")
    assert "gates passed" in out.stdout, out.stdout
    gdir = tmp_path / "graphs/syn"
    assert (gdir / "subgraph_hash.txt").exists() and (gdir / "degree_preserving_seed2/diagnostics.json").exists()
    for cond in ["real", "degree_preserving", "frozen"]:
        for k in [1, 2]:
            run("train.py", "c.yaml", "--condition", cond, "--seed", str(k), "--device", "cpu")
    assert (tmp_path / "checkpoints/smoke/syn/real_seed1.pt").exists()
    gens = (tmp_path / "runs/smoke/syn/real_seed1/generations.jsonl").read_text().splitlines()
    assert len(gens) == 2 and "ROMEO:" in json.loads(gens[0])["generations"]
    out = run("claim.py", "c.yaml", "--seeds", "1", "2")
    assert "mean Δ" in out.stdout
    run("evaluate.py", "c.yaml")
    assert "real" in (tmp_path / "results/scoreboard.md").read_text()
