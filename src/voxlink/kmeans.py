"""纯 Python k-means，用于把连续声学特征离散化为语音单元。

实现包含 k-means++ 初始化、Lloyd 迭代与空簇重初始化，全部由确定性
:class:`~voxlink._random.Rng` 驱动，保证可复现。
"""

from __future__ import annotations

from ._random import Rng
from .exceptions import ModelError
from .linalg import mean, sq_euclidean
from .types import Matrix

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


def _assign(data: Matrix, centers: Matrix) -> list[int]:
    """把每个点分配到最近的中心，返回标签。"""
    labels = []
    for x in data:
        best = 0
        best_d = sq_euclidean(x, centers[0])
        for j in range(1, len(centers)):
            d = sq_euclidean(x, centers[j])
            if d < best_d:
                best_d = d
                best = j
        labels.append(best)
    return labels


def _update(data: Matrix, labels: list[int], k: int, rng: Rng) -> Matrix:
    """根据分配重算中心；空簇用随机点重初始化。"""
    buckets: list[Matrix] = [[] for _ in range(k)]
    for x, lab in zip(data, labels, strict=True):
        buckets[lab].append(x)
    centers: Matrix = []
    for cluster in buckets:
        if cluster:
            centers.append(mean(cluster))
        else:
            # 空簇：从数据集中随机挑一个点，避免中心塌缩
            centers.append(list(data[rng.randint(0, len(data))]))
    return centers


def _inertia(data: Matrix, centers: Matrix, labels: list[int]) -> float:
    """簇内平方距离之和（越小越紧凑）。"""
    return sum(sq_euclidean(x, centers[lab]) for x, lab in zip(data, labels, strict=True))


class KMeans:
    """Lloyd 算法的 k-means 聚类器。

    Attributes:
        n_clusters: 簇数（即语音单元数）。
        centers: 拟合后的中心，未拟合时为 ``None``。
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        max_iters: int = 50,
        tol: float = 1e-4,
        seed: int = 0,
    ) -> None:
        if n_clusters < 1:
            raise ModelError("n_clusters 至少为 1")
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.centers: Matrix | None = None
        self.n_iter_: int = 0
        self.inertia_: float = 0.0

    def fit(self, data: Matrix) -> KMeans:
        """在 ``data``（形状 ``(n_samples, n_dims)``）上拟合。"""
        if len(data) < self.n_clusters:
            raise ModelError(f"样本数 {len(data)} 少于簇数 {self.n_clusters}")
        rng = Rng(self.seed)
        centers = _kmeans_plus_plus(data, self.n_clusters, rng)
        prev = float("inf")
        for it in range(1, self.max_iters + 1):
            labels = _assign(data, centers)
            centers = _update(data, labels, self.n_clusters, rng)
            inertia = _inertia(data, centers, labels)
            self.n_iter_ = it
            if abs(prev - inertia) <= self.tol:
                prev = inertia
                break
            prev = inertia
        self.centers = centers
        self.inertia_ = prev
        return self

    def predict(self, data: Matrix) -> list[int]:
        """给出每个样本的簇标签。"""
        if self.centers is None:
            raise ModelError("KMeans 尚未拟合")
        return _assign(data, self.centers)
