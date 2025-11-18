"""语言模型协议（占位，实现见后续提交）。

voxlink 把「大语言模型」抽象成一个最小接口：给定 token 前缀，返回下一个
token 的对数概率分布。参考实现是一个 n-gram 模型；真实系统可以用同样的
接口包一个基于 PyTorch 的解码器。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["LanguageModel"]


@runtime_checkable
class LanguageModel(Protocol):
    """自回归语言模型接口。"""

    @property
    def vocab_size(self) -> int: ...

    def logprobs(self, prefix: Sequence[int]) -> list[float]:
        """返回给定前缀下、整个词表的对数概率。"""
        ...
