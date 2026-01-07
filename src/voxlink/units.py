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

    def fit(self, feature_frames: Matrix) -> Quantizer:
        """在一批特征帧上训练码本。"""
        km = KMeans(
            self.config.n_units,
            max_iters=self.config.max_iters,
            tol=self.config.tol,
            seed=self.config.seed,
        )
        km.fit(feature_frames)
        self._kmeans = km
        return self

    def encode(self, feats: FeatureMatrix) -> list[int]:
        """把特征矩阵编码为离散单元序列。"""
        if self._kmeans is None:
            raise ModelError("Quantizer 尚未训练码本")
        units = self._kmeans.predict(feats.frames)
        if self.config.dedup:
            units = dedup_units(units)
        return units

    @property
    def centers(self) -> Matrix:
        if self._kmeans is None or self._kmeans.centers is None:
            raise ModelError("Quantizer 尚未训练码本")
        return self._kmeans.centers

    def save(self, path: str) -> None:
        """把码本与配置写成 JSON。"""
        write_json(
            path,
            {
                "config": self.config.to_dict(),
                "centers": self.centers,
            },
        )

    @classmethod
    def load(cls, path: str) -> Quantizer:
        """从 JSON 恢复量化器。"""
        payload = read_json(path)
        config = UnitConfig(**payload["config"])
        quant = cls(config)
        km = KMeans(config.n_units, seed=config.seed)
        km.centers = [list(map(float, row)) for row in payload["centers"]]
        quant._kmeans = km
        return quant
