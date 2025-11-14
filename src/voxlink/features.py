"""声学特征前端（占位，实现见后续提交）。"""

from __future__ import annotations

from .config import FrontendConfig
from .types import FeatureMatrix, Waveform

__all__ = ["Frontend"]


class Frontend:
    """把波形转成 log-mel / MFCC 特征。"""

    def __init__(self, config: FrontendConfig | None = None) -> None:
        self.config = config or FrontendConfig()

    def __call__(self, wave: Waveform) -> FeatureMatrix:  # pragma: no cover - 占位
        raise NotImplementedError
