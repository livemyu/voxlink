import pytest

from voxlink.config import ModelConfig
from voxlink.decode import beam_search, greedy_decode
from voxlink.ngram import NgramLM


def test_beam_and_greedy_agree_on_deterministic_chain() -> None:
    # 完全确定的链路上 beam 与 greedy 应给出相同结果
    lm = NgramLM(6, ModelConfig(order=3, smoothing=0.0001)).fit([[1, 2, 3, 4, 5, 0]] * 6)
    g = greedy_decode(lm, [1], max_new_tokens=10, eos=0)
    b = beam_search(lm, [1], max_new_tokens=10, eos=0, beam_size=5)
    assert g == b == [2, 3, 4, 5]


@pytest.mark.slow
def test_beam_scores_at_least_as_high_as_greedy() -> None:
    # beam search 的最优序列得分不应低于 greedy
    seqs = [[1, 2, 3, 0], [1, 3, 2, 0], [1, 2, 2, 0]]
    lm = NgramLM(5, ModelConfig(order=2, smoothing=0.1)).fit(seqs)
    g = greedy_decode(lm, [1], max_new_tokens=6, eos=0)
    b = beam_search(lm, [1], max_new_tokens=6, eos=0, beam_size=4)
    sg = lm.score([1, *g, 0])
    sb = lm.score([1, *b, 0])
    assert sb >= sg - 1e-9


def test_beam_length_penalty_runs() -> None:
    lm = NgramLM(5, ModelConfig(order=2)).fit([[1, 2, 3, 0]] * 3)
    out = beam_search(lm, [1], max_new_tokens=6, eos=0, beam_size=3, length_penalty=0.7)
    assert isinstance(out, list)
