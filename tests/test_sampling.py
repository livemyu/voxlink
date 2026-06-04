from voxlink._random import Rng
from voxlink.config import ModelConfig
from voxlink.decode import sample_decode, top_k_filter
from voxlink.ngram import NgramLM


def _lm() -> NgramLM:
    return NgramLM(6, ModelConfig(order=2, smoothing=0.01)).fit([[1, 2, 3, 4, 5, 0]] * 5)


def test_sample_decode_respects_max_tokens() -> None:
    lm = _lm()
    out = sample_decode(lm, [1], max_new_tokens=3, eos=99, rng=Rng(0))
    assert len(out) <= 3


def test_sample_decode_temperature_extremes() -> None:
    lm = _lm()
    # 低温接近贪心
    cold = sample_decode(lm, [1], 5, 0, rng=Rng(1), temperature=0.01)
    assert isinstance(cold, list)


def test_sample_decode_top_k() -> None:
    lm = _lm()
    out = sample_decode(lm, [1], 5, 0, rng=Rng(2), top_k=2)
    assert isinstance(out, list)


def test_top_k_filter_edge_cases() -> None:
    scores = [0.1, 0.2, 0.3]
    # k>=len 返回原样
    assert top_k_filter(scores, 5) == scores
    # k<=0 返回原样
    assert top_k_filter(scores, 0) == scores


def test_sample_decode_invalid_temperature() -> None:
    lm = _lm()
    try:
        sample_decode(lm, [1], 3, 0, rng=Rng(0), temperature=0.0)
    except ValueError:
        return
    raise AssertionError("temperature<=0 应抛 ValueError")
