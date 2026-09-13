import numpy as np

from flygpt.config import DataConfig
from flygpt.data import CharVocab, TokenStream, make_streams


def test_vocab_round_trips():
    v = CharVocab.from_text("hello world\n")
    assert v.decode(v.encode("hello world\n")) == "hello world\n"


def test_batches_are_contiguous_and_shifted():
    s = TokenStream(np.arange(1000), seq_len=8, batch_size=4)
    x, y = s.batch()
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert (y == x + 1).all()


def test_synthetic_streams_build():
    for task in ["repeat", "delayed_copy"]:
        vocab, tr, va = make_streams(DataConfig(task=task, seq_len=16, batch_size=4))
        assert len(vocab) > 1 and len(tr) > len(va) > 16
