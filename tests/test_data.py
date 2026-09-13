import numpy as np

from flygpt.data import CharVocab, TokenStream, reference_losses, split_text, corpus_hash


def test_vocab_round_trips():
    v = CharVocab.from_text("hello world\n")
    assert v.decode(v.encode("hello world\n")) == "hello world\n"


def test_batches_are_contiguous_and_shifted():
    s = TokenStream(np.arange(1000), seq_len=8, batch_size=4, seed=0)
    x, y = s.batch()
    assert x.shape == (4, 8) and (y == x + 1).all()


def test_same_seed_same_data_order():
    a = TokenStream(np.arange(1000), 8, 4, seed=7).batch()[0]
    b = TokenStream(np.arange(1000), 8, 4, seed=7).batch()[0]
    assert (a == b).all()


def test_split_and_hash_deterministic():
    text = "abc" * 1000
    tr, va, b = split_text(text, 0.9)
    assert b == 2700 and tr + va == text
    assert corpus_hash(text) == corpus_hash("abc" * 1000)


def test_bigram_beats_unigram_on_structured_text():
    rng = np.random.default_rng(0)
    ids = np.array([i % 5 for i in range(5000)])  # perfectly predictable bigrams
    r = reference_losses(ids[:4500], ids[4500:], 5)
    assert r["bigram_nats"] < r["unigram_nats"]
