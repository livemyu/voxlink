"""带回退的 n-gram 语言模型（参考实现）。

在离散 token 序列（语音单元 + 文本 token 的联合词表）上统计 n-gram 频次，
用 add-k 平滑估计条件概率，并在高阶上下文未见时逐级回退到低阶。它足够
轻量，可在测试中过拟合小语料，用来端到端验证「语音接入 LLM」的推理链路。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from .config import ModelConfig
from .exceptions import ModelError

__all__ = ["NgramLM"]

# 上下文 -> {下一个 token: 计数}
_CountTable = dict[tuple[int, ...], dict[int, int]]


class NgramLM:
    """add-k 平滑 + 回退的 n-gram 语言模型。"""

    def __init__(self, vocab_size: int, config: ModelConfig | None = None) -> None:
        if vocab_size < 1:
            raise ModelError("vocab_size 必须为正")
        self.config = config or ModelConfig()
        self._vocab_size = vocab_size
        # 对每个阶数保存一张计数表
        self._counts: list[_CountTable] = [defaultdict(dict) for _ in range(self.config.order)]
        self._totals: list[dict[tuple[int, ...], int]] = [
            defaultdict(int) for _ in range(self.config.order)
        ]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def order(self) -> int:
        return self.config.order

    def _observe(self, context: tuple[int, ...], token: int) -> None:
        """在对应阶数的表里累加一次 (context, token) 计数。"""
        n = len(context)
        table = self._counts[n]
        bucket = table[context]
        bucket[token] = bucket.get(token, 0) + 1
        self._totals[n][context] += 1

    def fit(self, sequences: Sequence[Sequence[int]]) -> NgramLM:
        """在多条 token 序列上统计各阶 n-gram。"""
        order = self.config.order
        for seq in sequences:
            tokens = list(seq)
            for i, token in enumerate(tokens):
                # 对当前位置，累加 0..order-1 阶的上下文
                for n in range(order):
                    if i - n < 0:
                        break
                    context = tuple(tokens[i - n : i])
                    self._observe(context, token)
        return self

    def _prob_at_order(self, context: tuple[int, ...], token: int) -> float | None:
        """在给定阶数上用 add-k 平滑估计 P(token | context)。

        若该上下文从未出现，返回 ``None`` 以触发回退。
        """
        n = len(context)
        total = self._totals[n].get(context, 0)
        if total == 0:
            return None
        k = self.config.smoothing
        count = self._counts[n].get(context, {}).get(token, 0)
        return (count + k) / (total + k * self._vocab_size)

    def prob(self, context: Sequence[int], token: int) -> float:
        """带回退的条件概率 P(token | context)。

        从最长可用上下文开始，逐级回退到更短上下文；全部落空时退化为均匀
        分布。
        """
        ctx = tuple(context)[-(self.config.order - 1) :] if self.config.order > 1 else ()
        while True:
            p = self._prob_at_order(ctx, token)
            if p is not None:
                return p
            if not ctx:
                return 1.0 / self._vocab_size
            ctx = ctx[1:]

    def _backoff_context(self, context: Sequence[int]) -> tuple[int, ...]:
        """选出最长的、曾出现过的上下文（用于构造归一化分布）。"""
        ctx = tuple(context)[-(self.config.order - 1) :] if self.config.order > 1 else ()
        while ctx and self._totals[len(ctx)].get(ctx, 0) == 0:
            ctx = ctx[1:]
        return ctx

    def logprobs(self, prefix: Sequence[int]) -> list[float]:
        """返回给定前缀下、整个词表的对数概率（已归一化）。"""
        ctx = self._backoff_context(prefix)
        n = len(ctx)
        total = self._totals[n].get(ctx, 0)
        k = self.config.smoothing
        denom = total + k * self._vocab_size
        counts = self._counts[n].get(ctx, {})
        if total == 0:
            uniform = -math.log(self._vocab_size)
            return [uniform] * self._vocab_size
        return [math.log((counts.get(t, 0) + k) / denom) for t in range(self._vocab_size)]

    def score(self, sequence: Sequence[int]) -> float:
        """序列的对数似然 sum log P(t_i | t_<i)。"""
        tokens = list(sequence)
        return math.fsum(math.log(self.prob(tokens[:i], tokens[i])) for i in range(len(tokens)))
