import numpy as np

from flygpt.config import GraphConfig
from flygpt.graph import build_graph, largest_weak_component, select_subgraph, synthetic_graph
from flygpt.scramble import scramble_edges


def test_synthetic_graph_has_no_self_loops_and_is_connected():
    src, dst = synthetic_graph(200, 0.02, seed=0)
    assert (src != dst).all()
    assert largest_weak_component(200, src, dst).all()


def test_io_nodes_disjoint_and_in_graph():
    g = build_graph(GraphConfig(source="synthetic", n_neurons=200, n_input=16, n_output=32))
    assert len(set(g.input_nodes) & set(g.output_nodes)) == 0
    assert g.input_nodes.max() < g.n and g.output_nodes.max() < g.n


def test_scramble_preserves_counts():
    src, dst = synthetic_graph(300, 0.03, seed=1)
    s2, d2 = scramble_edges(300, src, dst, seed=1)
    assert len(s2) == len(src)
    assert (s2 != d2).all()
    assert len(set(zip(s2.tolist(), d2.tolist()))) == len(src)  # no duplicates
    assert not (np.array_equal(s2, src) and np.array_equal(d2, dst))


def test_select_subgraph_relabels_and_is_connected():
    # fake "body ids": large sparse integers, like real connectome ids
    rng = np.random.default_rng(0)
    ids = rng.choice(10**9, 500, replace=False)
    src, dst = synthetic_graph(500, 0.02, seed=2)
    n, s, d, orig = select_subgraph(ids[src], ids[dst], n_target=100, seed=0)
    assert 0 < n <= 100
    assert s.max() < n and d.max() < n and s.min() >= 0
    assert len(orig) == n and set(orig) <= set(ids)
    assert largest_weak_component(n, s, d).all()
