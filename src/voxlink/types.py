"""跨模块共享的核心数据类型。

为了让纯 Python 内核不依赖 numpy，音频与特征统一用嵌套的 ``list[float]``
表示：波形是一维 ``Vector``，特征矩阵是二维 ``Matrix``（帧 × 维度）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

Vector = list[float]
Matrix = list[list[float]]


@dataclass(frozen=True)
class Waveform:
    """单声道 PCM 波形。

    Attributes:
        samples: 归一化到 ``[-1, 1]`` 的采样点。
        sample_rate: 采样率（Hz）。
    """

    samples: Vector
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate 必须为正整数")

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        """时长（秒）。"""
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class FeatureMatrix:
    """按帧排列的声学特征。

    Attributes:
        frames: 形状为 ``(num_frames, num_dims)`` 的特征。
        frame_rate: 每秒的帧数，用于对齐离散单元。
    """

    frames: Matrix
    frame_rate: float

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def num_dims(self) -> int:
        return len(self.frames[0]) if self.frames else 0


@dataclass
class Utterance:
    """一条训练 / 推理样本。"""

    audio: Waveform
    text: str = ""
    units: list[int] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
