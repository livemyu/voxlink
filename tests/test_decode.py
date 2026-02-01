from voxlink._random import Rng
from voxlink.config import ModelConfig
from voxlink.decode import (
    StopCriteria,
    beam_search,
    greedy_decode,
    sample_decode,
    top_k_filter,
)
from voxlink.ngram import NgramLM


def _toy_lm() -> NgramLM:
    # 稳定模式 1->2->3->4->EOS(0)
    return NgramLM(5, ModelConfig(order=2, smoothing=0.001)).fit([[1, 2, 3, 4, 0]] * 5)


def test_greedy_is_deterministic() -> None:
    lm = _toy_lm()
    a = greedy_decode(lm, [1], max_new_tokens=10, eos=0)
    b = greedy_decode(lm, [1], max_new_tokens=10, eos=0)
    assert a == b == [2, 3, 4]


def test_greedy_stops_at_eos() -> None:
    lm = _toy_lm()
    out = greedy_decode(lm, [1], max_new_tokens=10, eos=0)
    assert 0 not in out


def test_sample_decode_deterministic_with_seed() -> None:
    lm = _toy_lm()
    a = sample_decode(lm, [1], 8, 0, rng=Rng(3), temperature=0.7)
    b = sample_decode(lm, [1], 8, 0, rng=Rng(3), temperature=0.7)
    assert a == b


def test_top_k_filter_keeps_k() -> None:
    filtered = top_k_filter([0.1, 0.5, 0.2, 0.9], k=2)
    finite = [x for x in filtered if x != float("-inf")]
    assert len(finite) == 2


def test_beam_matches_greedy_on_toy() -> None:
    lm = _toy_lm()
    beam = beam_search(lm, [1], max_new_tokens=10, eos=0, beam_size=4)
    assert beam == [2, 3, 4]


def test_stop_criteria() -> None:
    sc = StopCriteria(eos=0, max_new_tokens=3, stop_tokens=[9])
    assert sc.is_stop_token(0)
    assert sc.is_stop_token(9)
    assert sc.reached_limit(3)
    assert not sc.reached_limit(2)
