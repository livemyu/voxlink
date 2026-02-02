import math

from voxlink._random import Rng
from voxlink.config import ModelConfig
from voxlink.ngram import NgramLM
from voxlink.vocab import SpecialTokens, build_joint_vocab


def test_logprobs_sum_to_one() -> None:
    lm = NgramLM(8, ModelConfig(order=2)).fit([[1, 2, 3, 4, 5, 6, 7]])
    probs = [math.exp(x) for x in lm.logprobs([2])]
    assert abs(sum(probs) - 1.0) < 1e-9


def test_ngram_overfits_toy_corpus() -> None:
    # 固定模式 1->2->3 反复出现，模型应把 3 排在 [1,2] 之后最高
    seqs = [[1, 2, 3, 1, 2, 3, 1, 2, 3]]
    lm = NgramLM(5, ModelConfig(order=3, smoothing=0.001)).fit(seqs)
    lp = lm.logprobs([1, 2])
    assert max(range(5), key=lambda i: lp[i]) == 3


def test_backoff_to_unigram_when_context_unseen() -> None:
    # 上下文 [3] 从未出现 -> 回退到一元分布（而非均匀）。
    lm = NgramLM(4, ModelConfig(order=2)).fit([[0, 1, 1, 1]])
    lp = lm.logprobs([3])
    # 训练中 token 1 出现最多，一元回退下应排第一
    assert max(range(4), key=lambda i: lp[i]) == 1


def test_uniform_without_training() -> None:
    # 完全未训练时，空上下文也未见 -> 均匀分布
    lm = NgramLM(4, ModelConfig(order=2))
    lp = lm.logprobs([])
    assert all(abs(x - lp[0]) < 1e-9 for x in lp)


def test_score_prefers_seen_sequence() -> None:
    lm = NgramLM(5, ModelConfig(order=2)).fit([[1, 2, 3, 4]] * 3)
    good = lm.score([1, 2, 3, 4])
    bad = lm.score([4, 3, 2, 1])
    assert good > bad


def test_sample_next_is_deterministic_with_seed() -> None:
    lm = NgramLM(6, ModelConfig(order=2)).fit([[1, 2, 3, 4, 5]] * 4)
    a = lm.sample_next([2], Rng(7))
    b = lm.sample_next([2], Rng(7))
    assert a == b


def test_joint_vocab_layout() -> None:
    v = build_joint_vocab(3, ["你好", "世界"], task_tokens=["<asr>"])
    assert SpecialTokens.PAD in v
    assert "u_0" in v and "u_2" in v
    assert "<asr>" in v
    assert "你好" in v
