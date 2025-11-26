"""voxlink 的配置对象。

所有配置都是带默认值的 frozen dataclass，方便序列化、比较与复现。
参数校验集中在 ``__post_init__``，非法值会抛出 :class:`ConfigError`。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .exceptions import ConfigError


@dataclass(frozen=True)
class FrontendConfig:
    """声学特征前端参数。"""

    sample_rate: int = 16000
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 80
    n_mfcc: int = 13
    fmin: float = 20.0
    fmax: float = 8000.0
    preemphasis: float = 0.97

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ConfigError("sample_rate 必须为正")
        if self.frame_shift_ms <= 0 or self.frame_length_ms <= 0:
            raise ConfigError("帧长与帧移必须为正")
        if self.n_fft < 2 or (self.n_fft & (self.n_fft - 1)) != 0:
            raise ConfigError("n_fft 必须是 >=2 的 2 的幂")
        if not 0 < self.n_mfcc <= self.n_mels:
            raise ConfigError("需满足 0 < n_mfcc <= n_mels")
        if not 0 <= self.preemphasis < 1:
            raise ConfigError("preemphasis 需落在 [0, 1)")

    @property
    def frame_length(self) -> int:
        return round(self.sample_rate * self.frame_length_ms / 1000.0)

    @property
    def frame_shift(self) -> int:
        return round(self.sample_rate * self.frame_shift_ms / 1000.0)

    @property
    def frame_rate(self) -> float:
        return 1000.0 / self.frame_shift_ms

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnitConfig:
    """离散语音单元（k-means 码本）参数。"""

    n_units: int = 100
    max_iters: int = 50
    tol: float = 1e-4
    seed: int = 0
    dedup: bool = True

    def __post_init__(self) -> None:
        if self.n_units < 2:
            raise ConfigError("n_units 至少为 2")
        if self.max_iters < 1:
            raise ConfigError("max_iters 至少为 1")
        if self.tol < 0:
            raise ConfigError("tol 不能为负")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """语言模型（n-gram 参考实现）参数。"""

    order: int = 3
    smoothing: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ConfigError("order 至少为 1")
        if self.smoothing <= 0:
            raise ConfigError("smoothing（add-k）必须为正")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
