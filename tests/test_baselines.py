import torch
from flygpt.baselines import build_baseline, match_hidden


def test_baselines_forward_and_param_matching():
    for kind in ["rnn", "gru", "transformer"]:
        h = match_hidden(kind, 20, 50_000)
        m = build_baseline(kind, 20, h)
        n = m.parameter_counts()["total"]
        assert 0.5 * 50_000 < n < 2 * 50_000, (kind, h, n)
        lg, _ = m(torch.randint(0, 20, (2, 8)))
        assert lg.shape == (2, 8, 20)
        assert m.generate(torch.tensor([[1, 2]]), 3).shape == (1, 5)
