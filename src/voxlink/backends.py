"""可选加速后端。

核心算子都有纯 Python 参考实现；当环境安装了 numpy 时，本模块提供等价、
更快的向量化版本，并通过 :func:`get_backend` 统一分发。没有 numpy 时一切
照常工作，只是更慢——这正是「离线优先、依赖可选」的设计。
"""

from __future__ import annotations

from . import linalg
from ._compat import has_numpy
from .types import Matrix

__all__ = [
    "Backend",
    "NumpyBackend",
    "PurePythonBackend",
    "available_backends",
    "get_backend",
]


class Backend:
    """算子后端接口。"""

    name = "base"

    def matmul(self, a: Matrix, b: Matrix) -> Matrix:  # pragma: no cover - 抽象
        raise NotImplementedError

    def pairwise_sq_dist(self, points: Matrix, centers: Matrix) -> Matrix:
        """每个点到每个中心的平方距离矩阵。"""
        raise NotImplementedError


class PurePythonBackend(Backend):
    """不依赖任何第三方库的参考后端。"""

    name = "python"

    def matmul(self, a: Matrix, b: Matrix) -> Matrix:
        return linalg.matmul(a, b)

    def pairwise_sq_dist(self, points: Matrix, centers: Matrix) -> Matrix:
        return [[linalg.sq_euclidean(p, c) for c in centers] for p in points]


class NumpyBackend(Backend):
    """基于 numpy 的向量化后端。"""

    name = "numpy"

    def __init__(self) -> None:
        import numpy as np

        self._np = np

    def matmul(self, a: Matrix, b: Matrix) -> Matrix:
        np = self._np
        result = np.asarray(a, dtype=float) @ np.asarray(b, dtype=float)
        return result.tolist()  # type: ignore[no-any-return]

    def pairwise_sq_dist(self, points: Matrix, centers: Matrix) -> Matrix:
        np = self._np
        p = np.asarray(points, dtype=float)
        c = np.asarray(centers, dtype=float)
        # ||p - c||^2 = ||p||^2 - 2 p·c + ||c||^2
        p2 = np.sum(p * p, axis=1, keepdims=True)
        c2 = np.sum(c * c, axis=1, keepdims=True).T
        dist = p2 - 2.0 * (p @ c.T) + c2
        return np.maximum(dist, 0.0).tolist()  # type: ignore[no-any-return]


def get_backend(prefer_numpy: bool = True) -> Backend:
    """返回可用的最佳后端。

    Args:
        prefer_numpy: 为 True 且 numpy 可用时返回 numpy 后端，否则纯 Python。
    """
    if prefer_numpy and has_numpy():
        return NumpyBackend()
    return PurePythonBackend()


def available_backends() -> list[str]:
    """列出当前环境可用的后端名。"""
    names = ["python"]
    if has_numpy():
        names.append("numpy")
    return names
