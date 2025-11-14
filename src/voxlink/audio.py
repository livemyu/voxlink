"""WAV 音频读写（占位，实现见后续提交）。"""

from __future__ import annotations

from .types import Waveform

__all__ = ["load_wav", "save_wav"]


def load_wav(path: str) -> Waveform:  # pragma: no cover - 占位
    raise NotImplementedError


def save_wav(path: str, wave: Waveform) -> None:  # pragma: no cover - 占位
    raise NotImplementedError
