"""离散余弦变换（DCT-II），用于从 log-mel 得到 MFCC。"""

from __future__ import annotations

import math
from functools import lru_cache

from .types import Matrix, Vector

__all__ = ["dct2", "dct2_matrix"]


@lru_cache(maxsize=16)
def dct2_matrix(n_out: int, n_in: int) -> tuple[tuple[float, ...], ...]:
    """预计算正交归一化的 DCT-II 基矩阵，形状 ``(n_out, n_in)``。"""
    basis = []
    for k in range(n_out):
        scale = math.sqrt(1.0 / n_in) if k == 0 else math.sqrt(2.0 / n_in)
        row = tuple(
            scale * math.cos(math.pi * (2 * n + 1) * k / (2 * n_in)) for n in range(n_in)
        )
        basis.append(row)
    return tuple(basis)


def dct2(vec: Vector, n_out: int) -> Vector:
    """对单个向量做 DCT-II，取前 ``n_out`` 个系数。"""
    mat = dct2_matrix(n_out, len(vec))
    return [math.fsum(b * x for b, x in zip(row, vec, strict=True)) for row in mat]


def dct2_rows(rows: Matrix, n_out: int) -> Matrix:
    """对矩阵每一行做 DCT-II。"""
    return [dct2(row, n_out) for row in rows]
