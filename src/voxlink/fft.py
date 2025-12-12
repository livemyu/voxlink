"""离散傅里叶变换：朴素 DFT 与迭代 radix-2 FFT。

纯 Python 复数实现。FFT 要求长度为 2 的幂；:func:`rfft_power` 直接给出
实信号的功率谱，供梅尔前端使用。
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache

from .types import Vector

__all__ = ["dft", "fft", "rfft_power"]


@lru_cache(maxsize=32)
def _twiddles(length: int) -> tuple[complex, ...]:
    """缓存某一级蝶形运算所需的旋转因子。"""
    step = -2.0j * math.pi / length
    half = length // 2
    return tuple(cmath.exp(step * k) for k in range(half))


def dft(signal: Vector) -> list[complex]:
    """朴素 O(n^2) 离散傅里叶变换（参考实现）。"""
    n = len(signal)
    out: list[complex] = []
    for k in range(n):
        acc = 0j
        for t, x in enumerate(signal):
            angle = -2.0 * math.pi * k * t / n
            acc += x * cmath.exp(1j * angle)
        out.append(acc)
    return out


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


def _bit_reverse(values: list[complex]) -> None:
    """就地位反转置换。"""
    n = len(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            values[i], values[j] = values[j], values[i]


def fft(signal: Vector) -> list[complex]:
    """迭代 radix-2 Cooley–Tukey FFT，要求长度为 2 的幂。"""
    n = len(signal)
    if not _is_pow2(n):
        raise ValueError(f"FFT 长度必须是 2 的幂，收到 {n}")
    values: list[complex] = [complex(x) for x in signal]
    _bit_reverse(values)

    length = 2
    while length <= n:
        half = length // 2
        tw = _twiddles(length)
        for start in range(0, n, length):
            for k in range(half):
                a = values[start + k]
                b = values[start + k + half] * tw[k]
                values[start + k] = a + b
                values[start + k + half] = a - b
        length *= 2
    return values
