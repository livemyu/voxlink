"""离散语音单元量化器（占位，实现见后续提交）。"""

from __future__ import annotations

from .config import UnitConfig
from .types import FeatureMatrix

__all__ = ["Quantizer"]


class Quantizer:
    """基于 k-means 码本，把连续特征映射为离散单元 id。"""

    def __init__(self, config: UnitConfig | None = None) -> None:
        self.config = config or UnitConfig()

    def encode(self, feats: FeatureMatrix) -> list[int]:  # pragma: no cover - 占位
        raise NotImplementedError
