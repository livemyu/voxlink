"""梅尔尺度与三角滤波器组。"""

from __future__ import annotations

import math

from .types import Matrix

__all__ = ["hz_to_mel", "mel_filterbank", "mel_to_hz"]


def hz_to_mel(hz: float) -> float:
    """HTK 风格的赫兹转梅尔。"""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    """梅尔转赫兹。"""
    return 700.0 * float(10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
) -> Matrix:
    """构造三角梅尔滤波器组。

    返回形状为 ``(n_mels, n_fft // 2 + 1)`` 的权重矩阵，可直接乘以功率谱。
    """
    if fmax > sample_rate / 2:
        fmax = sample_rate / 2.0
    n_bins = n_fft // 2 + 1

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    # n_mels 个滤波器需要 n_mels + 2 个等间隔的梅尔点
    mel_points = [mel_min + (mel_max - mel_min) * i / (n_mels + 1) for i in range(n_mels + 2)]
    hz_points = [mel_to_hz(m) for m in mel_points]
    # 映射到 FFT bin 索引
    bin_points = [math.floor((n_fft + 1) * hz / sample_rate) for hz in hz_points]
    # 保证 bin 单调递增：当梅尔点在低频过密时，相邻 bin 可能相等，
    # 强制每个中心至少比前一个大 1，避免出现全零滤波器。
    for i in range(1, len(bin_points)):
        if bin_points[i] <= bin_points[i - 1]:
            bin_points[i] = bin_points[i - 1] + 1

    fb: Matrix = [[0.0] * n_bins for _ in range(n_mels)]
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(left, center):
            if center > left and 0 <= k < n_bins:
                fb[m - 1][k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center and 0 <= k < n_bins:
                fb[m - 1][k] = (right - k) / (right - center)
    return fb
