"""基础数字信号处理：预加重、分帧、窗函数、重采样。

纯 Python 实现，作为特征前端的构件。
"""

from __future__ import annotations

import math

from .types import Matrix, Vector

__all__ = [
    "apply_window",
    "frame_signal",
    "hamming_window",
    "hann_window",
    "preemphasis",
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


def frame_signal(samples: Vector, frame_length: int, frame_shift: int) -> Matrix:
    """把一维信号切成重叠帧。

    末尾不足一帧的部分补零，保证至少产生一帧（信号非空时）。
    """
    if frame_length <= 0 or frame_shift <= 0:
        raise ValueError("frame_length 与 frame_shift 必须为正")
    n = len(samples)
    if n == 0:
        return []
    num_frames = 1 if n <= frame_length else 1 + math.ceil((n - frame_length) / frame_shift)

    frames: Matrix = []
    for i in range(num_frames):
        start = i * frame_shift
        chunk = samples[start : start + frame_length]
        if len(chunk) < frame_length:
            chunk = chunk + [0.0] * (frame_length - len(chunk))
        frames.append(chunk)
    return frames


def hann_window(size: int) -> Vector:
    """Hann 窗（周期性定义，适合 STFT）。"""
    if size < 1:
        raise ValueError("窗长必须为正")
    if size == 1:
        return [1.0]
    return [0.5 - 0.5 * math.cos(2.0 * math.pi * n / size) for n in range(size)]


def hamming_window(size: int) -> Vector:
    """Hamming 窗。"""
    if size < 1:
        raise ValueError("窗长必须为正")
    if size == 1:
        return [1.0]
    return [0.54 - 0.46 * math.cos(2.0 * math.pi * n / (size - 1)) for n in range(size)]


def apply_window(frame: Vector, window: Vector) -> Vector:
    """逐元素加窗。"""
    return [x * w for x, w in zip(frame, window, strict=True)]


def resample(samples: Vector, src_rate: int, dst_rate: int) -> Vector:
    """线性插值重采样。

    这是一个轻量参考实现，未做抗混叠滤波；对特征提取前的采样率对齐足够，
    对高保真场景应换用带 sinc 滤波的实现。
    """
    # TODO: 上采样时补一个低通滤波以抑制镜像分量
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("采样率必须为正")
    if src_rate == dst_rate or not samples:
        return list(samples)
    n_src = len(samples)
    n_dst = max(1, round(n_src * dst_rate / src_rate))
    ratio = (n_src - 1) / (n_dst - 1) if n_dst > 1 else 0.0
    out: Vector = []
    for i in range(n_dst):
        pos = i * ratio
        left = math.floor(pos)
        frac = pos - left
        if left + 1 < n_src:
            out.append(samples[left] * (1.0 - frac) + samples[left + 1] * frac)
        else:
            out.append(samples[left])
    return out
