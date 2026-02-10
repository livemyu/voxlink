from voxlink.eval import cer, corpus_metrics, levenshtein, wer


def test_levenshtein_basic() -> None:
    assert levenshtein(list("kitten"), list("sitting")) == 3
    assert levenshtein([], ["a", "b"]) == 2
    assert levenshtein(["a"], ["a"]) == 0


def test_wer_known_cases() -> None:
    assert wer("the cat sat", "the cat sat") == 0.0
    # 一个替换 / 三个词
    assert abs(wer("the cat sat", "the dog sat") - 1 / 3) < 1e-9
    # 完全不同
    assert wer("a b c", "x y z") == 1.0


def test_wer_empty_reference() -> None:
    assert wer("", "") == 0.0
    assert wer("", "hello") == 1.0


def test_cer_known_cases() -> None:
    assert cer("语音识别", "语音识别") == 0.0
    # 一个字错 / 四个字
    assert abs(cer("语音识别", "语音时别") - 1 / 4) < 1e-9


def test_corpus_metrics_aggregates() -> None:
    refs = ["the cat sat", "hello world"]
    hyps = ["the cat sat", "hello there"]
    m = corpus_metrics(refs, hyps)
    assert m.num_utts == 2
    assert m.ref_words == 5
    # 一个词错误 / 五个词
    assert abs(m.wer - 1 / 5) < 1e-9
