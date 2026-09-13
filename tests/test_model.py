import torch
import torch.nn.functional as F

from flygpt import Config, build_model
from flygpt.config import DataConfig, GraphConfig, ModelConfig
from flygpt.data import make_streams
from flygpt.graph import build_graph
from flygpt.model import FlyRNN


def small_model(vocab=10, **model_kw):
    g = build_graph(GraphConfig(source="synthetic", n_neurons=120, n_input=8, n_output=16))
    return FlyRNN(g, vocab, ModelConfig(embed_dim=8, **model_kw)), g


def test_forward_shapes():
    m, g = small_model()
    x = torch.randint(0, 10, (3, 7))
    logits, state = m(x)
    assert logits.shape == (3, 7, 10)
    assert state.shape == (3, g.n)


def test_gradients_reach_edge_weights_and_bias():
    m, _ = small_model()
    x = torch.randint(0, 10, (2, 5))
    logits, _ = m(x)
    logits.sum().backward()
    assert m.edge_weight.grad is not None and m.edge_weight.grad.abs().sum() > 0
    assert m.bias.grad is not None


def test_microsteps_and_leak_run():
    m, _ = small_model(microsteps=3, leak=0.5)
    logits, _ = m(torch.randint(0, 10, (2, 4)))
    assert torch.isfinite(logits).all()


def test_generate_shape():
    m, _ = small_model()
    out = m.generate(torch.tensor([[1, 2, 3]]), max_new=5)
    assert out.shape == (1, 8)


def test_loss_drops_on_repeated_sequence():
    torch.manual_seed(0)
    vocab, tr, _ = make_streams(DataConfig(task="repeat", seq_len=16, batch_size=8))
    m, _ = small_model(vocab=len(vocab))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)

    def loss_at():
        x, y = tr.batch()
        lg, _ = m(x)
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))

    first = loss_at().item()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_at()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first, (first, loss.item())


def test_build_model_all_types():
    for mt in ["fly", "rnn", "gpt"]:
        cfg = Config.from_dict({"model_type": mt, "graph": {"source": "synthetic", "n_neurons": 100, "n_input": 8, "n_output": 8},
                                "data": {"seq_len": 8}})
        m, _ = build_model(cfg, 20)
        lg, _ = m(torch.randint(0, 20, (2, 8)))
        assert lg.shape == (2, 8, 20)
