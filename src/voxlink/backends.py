"""可选加速后端。

核心算子都有纯 Python 参考实现；当环境安装了 numpy 时，本模块提供等价、
更快的向量化版本，并通过 :func:`get_backend` 统一分发。没有 numpy 时一切
照常工作，只是更慢——这正是「离线优先、依赖可选」的设计。
"""

from __future__ import annotations

from collections.abc import Sequence

from . import linalg
from ._compat import has_numpy
from .types import Matrix, Vector

__all__ = ["Backend", "PurePythonBackend", "get_backend"]


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
