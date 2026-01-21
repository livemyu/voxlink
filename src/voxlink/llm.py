"""语言模型协议。

voxlink 把「大语言模型」抽象成一个最小自回归接口：给定 token 前缀，返回下
一个 token 的对数概率分布。参考实现是一个带回退的 n-gram 模型
(:mod:`voxlink.ngram`)；真实系统可以用同样的协议包一个基于 PyTorch 的
解码器，从而复用 voxlink 的解码与评估设施。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

__all__ = ["LanguageModel"]


@runtime_checkable
class LanguageModel(Protocol):
    """自回归语言模型接口。"""

    @property
    def vocab_size(self) -> int:
        """词表大小（logprobs 的长度）。"""
        ...

    def logprobs(self, prefix: Sequence[int]) -> list[float]:
        """返回给定前缀下、整个词表的对数概率。"""
        ...
