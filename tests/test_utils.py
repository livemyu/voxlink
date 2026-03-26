import math

from voxlink.io_utils import read_jsonl, write_jsonl
from voxlink.mathx import log_softmax, logsumexp, softmax


def test_softmax_sums_to_one() -> None:
    p = softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1.0) < 1e-12


def test_softmax_numeric_stability() -> None:
    # 大数值不应溢出
    p = softmax([1000.0, 1000.0, 1000.0])
    assert all(abs(x - 1 / 3) < 1e-9 for x in p)


def test_softmax_temperature() -> None:
    sharp = softmax([1.0, 2.0], temperature=0.1)
    flat = softmax([1.0, 2.0], temperature=10.0)
    # 低温更尖锐
    assert max(sharp) > max(flat)


def test_logsumexp_matches_naive() -> None:
    xs = [0.1, 0.2, 0.3]
    naive = math.log(sum(math.exp(x) for x in xs))
    assert abs(logsumexp(xs) - naive) < 1e-9


def test_log_softmax_exp_is_softmax() -> None:
    xs = [0.5, 1.5, -0.5]
    lp = log_softmax(xs)
    p = softmax(xs)
    for a, b in zip(lp, p, strict=True):
        assert abs(math.exp(a) - b) < 1e-9


def test_jsonl_roundtrip(tmp_path) -> None:
    recs = [{"a": 1, "text": "你好"}, {"a": 2, "text": "世界"}]
    path = tmp_path / "x.jsonl"
    n = write_jsonl(str(path), recs)
    assert n == 2
    back = list(read_jsonl(str(path)))
    assert back == recs
