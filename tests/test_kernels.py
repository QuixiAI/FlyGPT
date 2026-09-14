"""The fused CUDA recurrence must reproduce the PyTorch sparse path: logits, final state, and every gradient."""
import pytest
import torch
import torch.nn.functional as F

from flygpt.config import ModelConfig, SequenceConfig
from flygpt.connectome import from_edge_table, select_io
from flygpt.connectome.synthetic import synthetic_edge_table
from flygpt.model import FlyRNN

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _model(backend, n=300, seed=0, init_scale=1.0):
    src, dst, w, _ = synthetic_edge_table(n=n, density=0.05, seed=seed)
    g = from_edge_table(src, dst, w)
    inp, out, _ = select_io(g, 24, "top_out_degree", 40, "top_in_degree")
    torch.manual_seed(seed)
    m = FlyRNN(g, inp, out, 65, ModelConfig(init_scale=init_scale, backend=backend), SequenceConfig(context=8, microsteps=2),
               edge_generator=torch.Generator().manual_seed(seed))
    return m.cuda()


@pytest.mark.parametrize("B", [4, 32, 40])   # partial warp, exact warp, two column chunks
def test_fused_matches_sparse(B):
    from flygpt.kernels import available
    if not available():
        pytest.skip("fused kernels could not be built (nvcc missing?)")
    ref, fused = _model("torch"), _model("cuda")
    fused.load_state_dict(ref.state_dict())
    torch.manual_seed(1)
    x = torch.randint(0, 65, (B, 8), device="cuda"); y = torch.randint(0, 65, (B, 8), device="cuda")
    outs = {}
    for name, m in [("ref", ref), ("fused", fused)]:
        assert m._use_kernels(x.device) == (name == "fused")
        lg, st = m(x)
        loss = F.cross_entropy(lg.reshape(-1, 65), y.reshape(-1))
        loss.backward()
        outs[name] = (lg.detach(), st.detach(), loss.item(), {n: p.grad.clone() for n, p in m.named_parameters()})
    lg0, st0, l0, g0 = outs["ref"]; lg1, st1, l1, g1 = outs["fused"]
    assert torch.allclose(lg0, lg1, atol=1e-5, rtol=1e-5), (lg0 - lg1).abs().max()
    assert torch.allclose(st0, st1, atol=1e-6, rtol=1e-5)
    assert abs(l0 - l1) < 1e-5
    for n in g0:
        assert torch.allclose(g0[n], g1[n], atol=1e-6 * max(g0[n].abs().max().item(), 1e-3), rtol=1e-4), n


def test_fused_gradients_reach_every_edge():
    from flygpt.kernels import available
    if not available():
        pytest.skip("fused kernels could not be built (nvcc missing?)")
    m = _model("cuda")
    x = torch.randint(0, 65, (8, 8), device="cuda")
    lg, _ = m(x)
    lg.float().pow(2).mean().backward()
    assert (m.edge_values.grad != 0).float().mean().item() > 0.99
