"""语音编码器协议（占位，实现见后续提交）。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import FeatureMatrix, Waveform

__all__ = ["SpeechEncoder"]


@runtime_checkable
class SpeechEncoder(Protocol):
    """把波形编码为帧级表示的统一接口。"""

    @property
    def output_dim(self) -> int: ...

    def encode(self, wave: Waveform) -> FeatureMatrix: ...
