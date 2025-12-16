"""基础数字信号处理：预加重、分帧、窗函数、重采样。

纯 Python 实现，作为特征前端的构件。
"""

from __future__ import annotations

import math

from .types import Matrix, Vector

__all__ = [
    "preemphasis",
    "frame_signal",
    "hann_window",
    "hamming_window",
    "resample",
]


def preemphasis(samples: Vector, coeff: float = 0.97) -> Vector:
    """一阶预加重滤波 ``y[n] = x[n] - a*x[n-1]``。"""
    if not samples:
        return []
    out = [samples[0]]
    for i in range(1, len(samples)):
        out.append(samples[i] - coeff * samples[i - 1])
    return out
