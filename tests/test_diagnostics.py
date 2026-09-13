import numpy as np

from flygpt.config import GateConfig
from flygpt.connectome.diagnostics import diagnose, io_path_lengths, path_gate
from flygpt.connectome.extract import EdgeGraph
from flygpt.connectome.synthetic import synthetic_graph


def test_path_lengths_on_a_chain():
    # 0->1->2->3 plus 3->0 so it's strongly connected
    g = EdgeGraph(4, np.array([0, 1, 2, 3]), np.array([1, 2, 3, 0]), np.ones(4, np.float32), np.arange(4))
    d = io_path_lengths(g, np.array([0]), np.array([1, 3]))
    assert list(d) == [1.0, 3.0]


def test_diagnose_fields_and_gate():
    g = synthetic_graph(200, 0.05, seed=0)
    inp, out = np.arange(10), np.arange(10, 40)
    diag = diagnose(g, inp, out)
    for k in ["largest_scc_fraction", "reciprocal_pairs", "io_reachable_fraction", "io_path_median", "io_path_p90", "io_path_max"]:
        assert k in diag
    assert 0 < diag["io_reachable_fraction"] <= 1
    ok = path_gate(diag, microsteps=2, gate=GateConfig(min_reachable_fraction=0.9, max_p90_chars=10))
    assert ok["passed"]
    bad = path_gate(diag, microsteps=1, gate=GateConfig(min_reachable_fraction=1.01, max_p90_chars=1))
    assert not bad["passed"]
