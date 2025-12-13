"""声学特征前端：STFT、log-mel 与 MFCC。

把波形转成按帧排列的特征矩阵。默认输出 MFCC；也可取 log-mel。
所有计算走纯 Python 内核（:mod:`voxlink.fft`、:mod:`voxlink.dsp`）。
"""

from __future__ import annotations

from .config import FrontendConfig
from .dsp import apply_window, frame_signal, hann_window, preemphasis
from .fft import rfft_power
from .types import FeatureMatrix, Matrix, Waveform

__all__ = ["Frontend", "power_spectrogram"]


def power_spectrogram(wave: Waveform, config: FrontendConfig) -> Matrix:
    """计算逐帧的功率谱。"""
    samples = preemphasis(wave.samples, config.preemphasis)
    frames = frame_signal(samples, config.frame_length, config.frame_shift)
    window = hann_window(config.frame_length)
    spec: Matrix = []
    for frame in frames:
        windowed = apply_window(frame, window)
        spec.append(rfft_power(windowed, config.n_fft))
    return spec
