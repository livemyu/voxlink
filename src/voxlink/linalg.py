"""纯 Python 线性代数辅助函数。

这些函数刻意保持朴素、无依赖，作为参考实现；当安装了 numpy 时，
:mod:`voxlink.backends` 会提供等价但更快的向量化版本。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = list[float]
Matrix = list[list[float]]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """两个等长向量的点积。"""
    if len(a) != len(b):
        raise ValueError(f"长度不匹配: {len(a)} vs {len(b)}")
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def add(a: Sequence[float], b: Sequence[float]) -> Vector:
    """逐元素相加。"""
    return [x + y for x, y in zip(a, b, strict=True)]


def sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    """逐元素相减。"""
    return [x - y for x, y in zip(a, b, strict=True)]


def scale(a: Sequence[float], factor: float) -> Vector:
    """向量数乘。"""
    return [x * factor for x in a]


def l2_norm(a: Sequence[float]) -> float:
    """向量的 L2 范数。"""
    return math.sqrt(math.fsum(x * x for x in a))


def sq_euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """平方欧氏距离（避免开方，用于聚类比较）。"""
    return math.fsum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def mean(vectors: Sequence[Sequence[float]]) -> Vector:
    """对一组向量求均值。"""
    if not vectors:
        raise ValueError("不能对空集合求均值")
    n = len(vectors)
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    return [s / n for s in acc]
