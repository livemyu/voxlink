from voxlink.data import WORDS, synth_corpus, synth_utterance
from voxlink.dataset import train_test_split


def test_synth_word_is_deterministic() -> None:
    a = synth_utterance(["one", "two"])
    b = synth_utterance(["one", "two"])
    assert a.audio.samples == b.audio.samples
    assert a.text == "one two"


def test_synth_words_differ() -> None:
    a = synth_utterance(["one"]).audio.samples
    b = synth_utterance(["five"]).audio.samples
    assert a != b


def test_synth_corpus_length() -> None:
    corpus = synth_corpus([["one"], ["two", "three"]])
    assert len(corpus) == 2
    assert corpus[1].text == "two three"


def test_word_bank_covers_expected() -> None:
    for w in ("zero", "one", "yes", "no", "up", "down"):
        assert w in WORDS


def test_train_test_split_is_deterministic() -> None:
    utts = synth_corpus([[w] for w in WORDS])
    a_train, a_test = train_test_split(utts, test_ratio=0.3, seed=5)
    _, b_test = train_test_split(utts, test_ratio=0.3, seed=5)
    assert [u.text for u in a_test] == [u.text for u in b_test]
    assert len(a_train) + len(a_test) == len(utts)
