import numpy as np
import torch
import torch.nn.functional as F

from flygpt.config import ModelConfig, SequenceConfig
from flygpt.connectome.interface import select_io
from flygpt.connectome.synthetic import synthetic_graph
from flygpt.data import TokenStream
from flygpt.model import FlyRNN, make_frozen_fly, spectral_radius


def small(vocab=10, n=120, microsteps=2, seed=0, **mk):
    g = synthetic_graph(n, 0.05, seed=seed)
    inp, out, _ = select_io(g, 8, "top_out_degree", 16, "top_in_degree")
    m = FlyRNN(g, inp, out, vocab, ModelConfig(embed_dim=8, **mk), SequenceConfig(context=16, microsteps=microsteps),
               edge_generator=torch.Generator().manual_seed(seed))
    return m, g


def test_forward_shapes():
    m, g = small()
    lg, st = m(torch.randint(0, 10, (3, 7)))
    assert lg.shape == (3, 7, 10) and st.shape == (3, g.n)


def test_sparse_matches_dense_reference():
    m, g = small()
    state = torch.randn(4, g.n) * 0.5
    x = torch.randint(0, 10, (4,))
    a = m.step(state, x)
    b = m.dense_reference_step(state, x)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_gradients_reach_every_edge_value_and_leak():
    m, _ = small()
    lg, _ = m(torch.randint(0, 10, (2, 6)))
    lg.sum().backward()
    assert m.edge_values.grad is not None and (m.edge_values.grad != 0).all(), "some edges receive no gradient"
    assert m.raw_leak.grad is not None and m.bias.grad is not None


def test_leak_init_and_degree_normalization():
    m, g = small()
    assert torch.allclose(m.leak, torch.full((g.n,), 0.5))
    indeg = torch.bincount(m.indices[0], minlength=g.n).float()
    assert torch.allclose(m.edge_scale, 1 / indeg[m.indices[0]].sqrt())
    m2, _ = small(degree_normalization=False)
    assert (m2.edge_scale == 1).all()


def test_paired_seed_gives_identical_adapters_and_edge_stream():
    torch.manual_seed(1); a, _ = small(seed=3)
    torch.manual_seed(1); b, _ = small(seed=3)
    assert torch.equal(a.embed.weight, b.embed.weight) and torch.equal(a.edge_values, b.edge_values)


def test_frozen_fly_only_trains_adapters_and_has_target_radius():
    g = synthetic_graph(120, 0.05, seed=0)
    inp, out, _ = select_io(g, 8, "top_out_degree", 16, "top_in_degree")
    m = make_frozen_fly(g, inp, out, 10, ModelConfig(embed_dim=8), SequenceConfig(microsteps=2), seed=0)
    assert m.recurrent_parameters() == []
    assert abs(spectral_radius(m) - 0.95) < 0.1
    assert set(m.edge_values.abs().unique().tolist()).__len__() == 1  # +-c


def test_loss_drops_on_repeated_sequence():
    torch.manual_seed(0)
    text = ("abcdefgh" * 400)
    vocab = sorted(set(text)); ids = np.array([vocab.index(c) for c in text])
    tr = TokenStream(ids, 16, 8, seed=0)
    m, _ = small(vocab=len(vocab))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    def loss_at():
        x, y = tr.batch(); lg, _ = m(x)
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
    first = loss_at().item()
    for _ in range(150):
        opt.zero_grad(); l = loss_at(); l.backward(); opt.step()
    assert l.item() < 0.5 * first, (first, l.item())


def test_generate_and_states():
    m, g = small()
    out, states = m.generate(torch.tensor([[1, 2, 3]]), 5, return_states=True)
    assert out.shape == (1, 8) and states.shape == (8, g.n)
