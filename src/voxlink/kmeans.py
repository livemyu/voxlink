"""纯 Python k-means，用于把连续声学特征离散化为语音单元。

实现包含 k-means++ 初始化、Lloyd 迭代与空簇重初始化，全部由确定性
:class:`~voxlink._random.Rng` 驱动，保证可复现。
"""

from __future__ import annotations

from .exceptions import ModelError
from .linalg import mean, sq_euclidean
from .types import Matrix, Vector
from ._random import Rng

__all__ = ["KMeans"]


def _kmeans_plus_plus(data: Matrix, k: int, rng: Rng) -> Matrix:
    """k-means++ 选取初始中心，让初始中心尽量分散。"""
    n = len(data)
    first = rng.randint(0, n)
    centers: Matrix = [list(data[first])]
    while len(centers) < k:
        # 每个点到最近中心的平方距离
        dists = [min(sq_euclidean(x, c) for c in centers) for x in data]
        total = sum(dists)
        if total <= 0.0:
            # 所有点重合，随机补齐
            centers.append(list(data[rng.randint(0, n)]))
            continue
        target = rng.random() * total
        acc = 0.0
        chosen = n - 1
        for i, d in enumerate(dists):
            acc += d
            if acc >= target:
                chosen = i
                break
        centers.append(list(data[chosen]))
    return centers
