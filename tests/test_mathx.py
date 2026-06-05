import math

import pytest

from voxlink.mathx import log_softmax, logsumexp, sample_categorical, softmax


def test_logsumexp_empty_and_neg_inf() -> None:
    assert logsumexp([]) == -math.inf
    assert logsumexp([-math.inf, -math.inf]) == -math.inf


def test_softmax_invalid_temperature() -> None:
    with pytest.raises(ValueError):
        softmax([1.0, 2.0], temperature=0.0)


def test_log_softmax_sums_via_exp() -> None:
    lp = log_softmax([0.0, 0.0, 0.0])
    assert abs(sum(math.exp(x) for x in lp) - 1.0) < 1e-9


def test_sample_categorical_picks_by_cumsum() -> None:
    probs = [0.2, 0.3, 0.5]
    assert sample_categorical(probs, 0.0) == 0
    assert sample_categorical(probs, 0.25) == 1
    assert sample_categorical(probs, 0.9) == 2


def test_sample_categorical_saturates() -> None:
    # u 超过总和时返回最后一个下标
    assert sample_categorical([0.1, 0.1], 1.0) == 1
