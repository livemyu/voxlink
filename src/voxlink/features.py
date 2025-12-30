"""声学特征前端：STFT、log-mel 与 MFCC。

把波形转成按帧排列的特征矩阵。默认输出 MFCC；也可取 log-mel。
所有计算走纯 Python 内核（:mod:`voxlink.fft`、:mod:`voxlink.dsp`）。
"""

from __future__ import annotations

import math

from .config import FrontendConfig
from .dct import dct2_rows
from .dsp import apply_window, frame_signal, hann_window, preemphasis
from .fft import rfft_power
from .linalg import matmul, transpose
from .melbank import mel_filterbank
from .types import FeatureMatrix, Matrix, Waveform

__all__ = ["Frontend", "cmvn", "log_mel_spectrogram", "mfcc", "power_spectrogram"]

_LOG_FLOOR = 1e-10


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


def log_mel_spectrogram(wave: Waveform, config: FrontendConfig) -> Matrix:
    """对功率谱应用梅尔滤波器组并取对数。

    返回形状 ``(num_frames, n_mels)``。
    """
    power = power_spectrogram(wave, config)
    if not power:
        return []
    fb = mel_filterbank(config.n_mels, config.n_fft, config.sample_rate, config.fmin, config.fmax)
    # (frames, bins) @ (bins, n_mels) -> (frames, n_mels)
    mel = matmul(power, transpose(fb))
    return [[math.log(max(v, _LOG_FLOOR)) for v in row] for row in mel]


def mfcc(wave: Waveform, config: FrontendConfig) -> Matrix:
    """从 log-mel 通过 DCT-II 得到 MFCC，返回 ``(num_frames, n_mfcc)``。"""
    logmel = log_mel_spectrogram(wave, config)
    if not logmel:
        return []
    return dct2_rows(logmel, config.n_mfcc)


def cmvn(feats: Matrix, *, variance: bool = False) -> Matrix:
    """倒谱均值（方差）归一化。

    对每一维减去均值；``variance=True`` 时再除以标准差。
    """
    if not feats:
        return []
    n = len(feats)
    dim = len(feats[0])
    means = [math.fsum(row[j] for row in feats) / n for j in range(dim)]
    if variance:
        stds = []
        for j in range(dim):
            var = math.fsum((row[j] - means[j]) ** 2 for row in feats) / n
            stds.append(math.sqrt(var) or 1.0)
    else:
        stds = [1.0] * dim
    return [[(row[j] - means[j]) / stds[j] for j in range(dim)] for row in feats]


class Frontend:
    """把波形转成特征矩阵的可调用前端。

    Args:
        config: 前端配置；缺省使用默认值。
        feature: ``"mfcc"``（默认）或 ``"logmel"``。
        normalize: 是否做倒谱均值归一化。
    """

    def __init__(
        self,
        config: FrontendConfig | None = None,
        *,
        feature: str = "mfcc",
        normalize: bool = True,
    ) -> None:
        if feature not in ("mfcc", "logmel"):
            raise ValueError(f"未知特征类型: {feature}")
        self.config = config or FrontendConfig()
        self.feature = feature
        self.normalize = normalize

    @property
    def output_dim(self) -> int:
        return self.config.n_mfcc if self.feature == "mfcc" else self.config.n_mels

    def __call__(self, wave: Waveform) -> FeatureMatrix:
        if self.feature == "mfcc":
            frames = mfcc(wave, self.config)
        else:
            frames = log_mel_spectrogram(wave, self.config)
        if self.normalize:
            frames = cmvn(frames)
        return FeatureMatrix(frames=frames, frame_rate=self.config.frame_rate)
