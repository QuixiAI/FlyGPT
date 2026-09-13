import numpy as np

from flygpt.config import GraphConfig
from flygpt.connectome.extract import from_edge_table, extract_dense_core, largest_scc, directed_core, save_subgraph, load_subgraph
from flygpt.connectome.synthetic import synthetic_edge_table


def test_from_edge_table_relabels_and_filters_candidates():
    s, d, w, regions = synthetic_edge_table(300, 0.02, seed=0)
    cand = np.array([i for i, r in regions.items() if r == "central_brain"])
    g = from_edge_table(s, d, w, candidate_ids=cand)
    assert set(g.ids) <= set(cand)
    assert g.src.max() < g.n and g.dst.max() < g.n and (g.src != g.dst).all()


def test_directed_core_monotone():
    s, d, w, _ = synthetic_edge_table(300, 0.03, seed=1)
    g = from_edge_table(s, d, w)
    sizes = [directed_core(g, k).sum() for k in range(0, 8)]
    assert all(a >= b for a, b in zip(sizes, sizes[1:]))


def test_dense_core_hits_target_and_is_strongly_connected():
    s, d, w, _ = synthetic_edge_table(600, 0.03, seed=2)
    g = from_edge_table(s, d, w)
    core, log = extract_dense_core(g, target=150, min_synapses=1)
    assert 150 <= core.n <= 160, (core.n, log)
    assert largest_scc(core).all()
    assert (core.weight >= 1).all() and log["core_k"] >= 1


def test_min_synapses_threshold_removes_edges():
    s, d, w, _ = synthetic_edge_table(400, 0.03, seed=3)
    g = from_edge_table(s, d, w)
    core3, log3 = extract_dense_core(g, target=100, min_synapses=3)
    assert (core3.weight >= 3).all()
    assert log3["edges_after_threshold"] < g.n_edges


def test_extraction_is_deterministic():
    s, d, w, _ = synthetic_edge_table(500, 0.03, seed=4)
    g = from_edge_table(s, d, w)
    a, _ = extract_dense_core(g, 120, 1)
    b, _ = extract_dense_core(g, 120, 1)
    assert a.hash() == b.hash()


def test_save_and_load_round_trip(tmp_path):
    s, d, w, _ = synthetic_edge_table(300, 0.03, seed=5)
    core, _ = extract_dense_core(from_edge_table(s, d, w), 80, 1)
    stats = save_subgraph(core, GraphConfig(target_neurons=80, min_synapses=1), tmp_path)
    g2 = load_subgraph(tmp_path / "subgraph_edges.pt")
    assert g2.hash() == core.hash() == stats["hash"]
    assert (tmp_path / "subgraph_node_ids.txt").exists() and (tmp_path / "subgraph_hash.txt").exists()
