"""离散语音单元量化器。

把连续声学特征（来自 :class:`~voxlink.features.Frontend`）通过 k-means
码本映射为离散单元 id 序列——这是「离散语音 token 接入 LLM」的基础。
可选地对相邻重复单元做游程去重（run-length dedup）。
"""

from __future__ import annotations

from .config import UnitConfig
from .exceptions import ModelError
from .io_utils import read_json, write_json
from .kmeans import KMeans
from .types import FeatureMatrix, Matrix

__all__ = ["Quantizer", "dedup_units"]


def dedup_units(units: list[int]) -> list[int]:
    """折叠相邻的重复单元，例如 ``[3,3,5,5,5,1] -> [3,5,1]``。"""
    out: list[int] = []
    for u in units:
        if not out or out[-1] != u:
            out.append(u)
    return out


class Quantizer:
    """基于 k-means 码本，把连续特征映射为离散单元 id。"""

    def __init__(self, config: UnitConfig | None = None) -> None:
        self.config = config or UnitConfig()
        self._kmeans: KMeans | None = None

    @property
    def n_units(self) -> int:
        return self.config.n_units

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None and self._kmeans.centers is not None
