"""WAV 音频读写。

只依赖标准库 :mod:`wave` 与 :mod:`struct`，支持 8/16/24/32 位 PCM，
读入后统一转成 ``[-1, 1]`` 的浮点单声道波形（多声道取平均）。
"""

from __future__ import annotations

import struct
import wave

from .exceptions import AudioError
from .types import Vector, Waveform

__all__ = ["load_wav", "save_wav", "pcm_to_float", "float_to_pcm"]

_INT_MAX = {8: 127, 16: 32767, 24: 8388607, 32: 2147483647}


def pcm_to_float(samples: list[int], bits: int) -> Vector:
    """把整数 PCM 采样归一化到 ``[-1, 1]``。"""
    if bits not in _INT_MAX:
        raise AudioError(f"不支持的位深: {bits}")
    peak = _INT_MAX[bits]
    return [max(-1.0, s / peak) for s in samples]


def float_to_pcm(samples: Vector, bits: int) -> list[int]:
    """把 ``[-1, 1]`` 的浮点采样量化回整数 PCM（带饱和裁剪）。"""
    if bits not in _INT_MAX:
        raise AudioError(f"不支持的位深: {bits}")
    peak = _INT_MAX[bits]
    out = []
    for s in samples:
        clipped = min(1.0, max(-1.0, s))
        out.append(int(round(clipped * peak)))
    return out
