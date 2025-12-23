"""梅尔尺度与三角滤波器组。"""

from __future__ import annotations

import math

from .types import Matrix

__all__ = ["hz_to_mel", "mel_to_hz", "mel_filterbank"]


def hz_to_mel(hz: float) -> float:
    """HTK 风格的赫兹转梅尔。"""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """梅尔转赫兹。"""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
