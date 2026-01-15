"""模态适配器：把语音侧表示对齐到语言模型输入空间。

包含两条常见路径：

- :class:`LinearProjector` —— 对连续帧表示做线性投影（可选偏置）。
- :class:`UnitEmbedding` —— 为离散语音单元查表得到嵌入。

两者都用确定性种子初始化权重，保证可复现。
"""

from __future__ import annotations

from ._random import Rng
from .exceptions import ModelError
from .linalg import matvec
from .types import Matrix, Vector

__all__ = ["LinearProjector", "UnitEmbedding"]


class LinearProjector:
    """把 ``in_dim`` 维帧表示线性投影到 ``out_dim`` 维。"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool = True,
        seed: int = 0,
    ) -> None:
        if in_dim < 1 or out_dim < 1:
            raise ModelError("投影维度必须为正")
        self.in_dim = in_dim
        self.out_dim = out_dim
        rng = Rng(seed)
        # Xavier 风格的小尺度初始化
        scale = (6.0 / (in_dim + out_dim)) ** 0.5
        self.weight: Matrix = [
            [(rng.random() * 2 - 1) * scale for _ in range(in_dim)] for _ in range(out_dim)
        ]
        self.bias: Vector = [0.0] * out_dim if bias else []

    def project_frame(self, frame: Vector) -> Vector:
        """投影单帧。"""
        out = matvec(self.weight, frame)
        if self.bias:
            out = [o + b for o, b in zip(out, self.bias, strict=True)]
        return out

    def __call__(self, frames: Matrix) -> Matrix:
        return [self.project_frame(f) for f in frames]


class UnitEmbedding:
    """离散语音单元的可查表嵌入。

    每个单元 id 对应一行 ``dim`` 维向量，用确定性种子随机初始化。
    """

    def __init__(self, n_units: int, dim: int, *, seed: int = 0) -> None:
        if n_units < 1 or dim < 1:
            raise ModelError("n_units 与 dim 必须为正")
        self.n_units = n_units
        self.dim = dim
        rng = Rng(seed)
        scale = (1.0 / dim) ** 0.5
        self.table: Matrix = [
            [rng.gauss(0.0, scale) for _ in range(dim)] for _ in range(n_units)
        ]

    def lookup(self, unit: int) -> Vector:
        if not 0 <= unit < self.n_units:
            raise ModelError(f"单元 id 越界: {unit}")
        return list(self.table[unit])

    def __call__(self, units: list[int]) -> Matrix:
        return [self.lookup(u) for u in units]
