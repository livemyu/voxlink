"""数值稳定的数学工具（softmax、logsumexp 等）。"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = list[float]


def logsumexp(scores: Sequence[float]) -> float:
    """数值稳定的 ``log(sum(exp(scores)))``。"""
    if not scores:
        return -math.inf
    m = max(scores)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.fsum(math.exp(s - m) for s in scores))


def softmax(scores: Sequence[float], temperature: float = 1.0) -> Vector:
    """带温度的 softmax，返回归一化概率分布。"""
    if temperature <= 0:
        raise ValueError("temperature 必须为正")
    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = math.fsum(exps)
    return [e / total for e in exps]


def log_softmax(scores: Sequence[float]) -> Vector:
    """对数域 softmax。"""
    lse = logsumexp(scores)
    return [s - lse for s in scores]
