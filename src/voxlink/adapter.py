"""模态适配器（占位，实现见后续提交）。

把语音侧的表示（编码器输出或离散单元的嵌入）投影 / 对齐到语言模型的
输入空间，是「语音接入 LLM」的关键一环。
"""

from __future__ import annotations

from .types import Matrix

__all__ = ["LinearProjector"]


class LinearProjector:  # pragma: no cover - 占位
    """把 ``in_dim`` 的帧表示线性投影到 ``out_dim``。"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __call__(self, frames: Matrix) -> Matrix:
        raise NotImplementedError
