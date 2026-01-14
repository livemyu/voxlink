"""语音编码器。

参考实现 :class:`MFCCEncoder` 直接复用特征前端，并在其上做可选的帧下采样
（stacking / subsampling）与 L2 归一化，输出帧级表示。它满足
:class:`SpeechEncoder` 协议，可与适配器、量化器组合。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import FrontendConfig
from .features import Frontend
from .types import FeatureMatrix, Matrix, Vector, Waveform

__all__ = ["SpeechEncoder", "MFCCEncoder", "mean_pool", "max_pool", "stack_frames"]


@runtime_checkable
class SpeechEncoder(Protocol):
    """把波形编码为帧级表示的统一接口。"""

    @property
    def output_dim(self) -> int: ...

    def encode(self, wave: Waveform) -> FeatureMatrix: ...


def mean_pool(frames: Matrix) -> Vector:
    """对所有帧做均值池化，得到一条句级向量。"""
    if not frames:
        return []
    n = len(frames)
    dim = len(frames[0])
    acc = [0.0] * dim
    for row in frames:
        for j, v in enumerate(row):
            acc[j] += v
    return [s / n for s in acc]


def max_pool(frames: Matrix) -> Vector:
    """对所有帧做逐维最大池化。"""
    if not frames:
        return []
    dim = len(frames[0])
    out = list(frames[0])
    for row in frames[1:]:
        for j in range(dim):
            if row[j] > out[j]:
                out[j] = row[j]
    return out


def stack_frames(frames: Matrix, factor: int) -> Matrix:
    """把连续 ``factor`` 帧拼接成一帧，实现时间下采样。

    这是「长度适配」的一种常见做法：语音帧率远高于文本 token 率，拼帧能把
    序列缩短到语言模型更易处理的长度。末尾不足的帧丢弃。
    """
    if factor < 1:
        raise ValueError("factor 必须为正")
    if factor == 1:
        return [list(row) for row in frames]
    out: Matrix = []
    for i in range(0, len(frames) - factor + 1, factor):
        merged: Vector = []
        for k in range(factor):
            merged.extend(frames[i + k])
        out.append(merged)
    return out
