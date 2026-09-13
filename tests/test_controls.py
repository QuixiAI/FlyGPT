import numpy as np

from flygpt.connectome.controls import degree_preserving_rewire, uniform_random_rewire, plateau_agrees, build_control
from flygpt.connectome.interface import select_io
from flygpt.connectome.synthetic import synthetic_graph


def _no_loops_no_dups(g):
    assert (g.src != g.dst).all()
    assert len(set(zip(g.src.tolist(), g.dst.tolist()))) == g.n_edges


def test_degree_preserving_preserves_degrees_and_counts():
    g = synthetic_graph(200, 0.05, seed=0)
    r = degree_preserving_rewire(g, seed=1)
    h = r.graph
    assert h.n == g.n and h.n_edges == g.n_edges
    assert np.array_equal(h.in_degree(), g.in_degree()) and np.array_equal(h.out_degree(), g.out_degree())
    assert np.array_equal(np.sort(h.weight), np.sort(g.weight))   # §4.2 weight multiset preserved
    assert np.array_equal(h.ids, g.ids)
    _no_loops_no_dups(h)
    assert r.accepted_swaps >= 10 * g.n_edges
    assert r.survival < 0.5


def test_two_shuffles_agree_on_survival():
    g = synthetic_graph(200, 0.05, seed=0)
    a, b = degree_preserving_rewire(g, 1), degree_preserving_rewire(g, 2)
    assert plateau_agrees(a, b, tol=0.05)
    assert a.survival_history[-1] < a.survival_history[0]          # went down overall
    assert abs(a.survival_history[-1] - a.survival_history[-2]) < 1e-3  # and plateaued


def test_uniform_random_preserves_counts_only():
    g = synthetic_graph(200, 0.05, seed=0)
    h = uniform_random_rewire(g, seed=3)
    assert h.n == g.n and h.n_edges == g.n_edges
    _no_loops_no_dups(h)
    assert not np.array_equal(h.in_degree(), g.in_degree())


def test_io_sets_identical_across_real_and_degree_preserving():
    g = synthetic_graph(200, 0.05, seed=0)
    h, _ = build_control(g, "degree_preserving", seed=5)
    a = select_io(g, 16, "top_out_degree", 32, "top_in_degree")
    b = select_io(h, 16, "top_out_degree", 32, "top_in_degree")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_rewire_is_seed_deterministic():
    g = synthetic_graph(150, 0.05, seed=0)
    a, b = degree_preserving_rewire(g, 9).graph, degree_preserving_rewire(g, 9).graph
    assert np.array_equal(a.src, b.src) and np.array_equal(a.dst, b.dst)
