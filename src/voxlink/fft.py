"""离散傅里叶变换：朴素 DFT 与迭代 radix-2 FFT。

纯 Python 复数实现。FFT 要求长度为 2 的幂；:func:`rfft_power` 直接给出
实信号的功率谱，供梅尔前端使用。
"""

from __future__ import annotations

import cmath
import math

from .types import Vector

__all__ = ["dft", "fft", "rfft_power"]


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
