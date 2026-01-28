"""带回退的 n-gram 语言模型（参考实现）。

在离散 token 序列（语音单元 + 文本 token 的联合词表）上统计 n-gram 频次，
用 add-k 平滑估计条件概率，并在高阶上下文未见时逐级回退到低阶。它足够
轻量，可在测试中过拟合小语料，用来端到端验证「语音接入 LLM」的推理链路。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from ._random import Rng
from .config import ModelConfig
from .exceptions import ModelError
from .io_utils import read_json, write_json
from .mathx import softmax

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

    def sample_next(self, prefix: Sequence[int], rng: Rng, temperature: float = 1.0) -> int:
        """从下一 token 分布中采样（带温度）。"""
        logp = self.logprobs(prefix)
        if temperature != 1.0:
            logp = [lp / temperature for lp in logp]
        probs = softmax(logp)
        target = rng.random()
        acc = 0.0
        for token, p in enumerate(probs):
            acc += p
            if acc >= target:
                return token
        return len(probs) - 1

    def to_dict(self) -> dict[str, object]:
        """把模型序列化为可 JSON 化的字典。"""
        # 上下文元组用空格连接的字符串作为 key（空上下文为 ""）
        tables = []
        for table in self._counts:
            flat = {
                " ".join(map(str, ctx)): dict(bucket.items()) for ctx, bucket in table.items()
            }
            tables.append(flat)
        return {
            "vocab_size": self._vocab_size,
            "config": self.config.to_dict(),
            "counts": tables,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> NgramLM:
        """从 :meth:`to_dict` 的结果重建模型。"""
        config = ModelConfig(**payload["config"])  # type: ignore[arg-type]
        model = cls(int(payload["vocab_size"]), config)  # type: ignore[arg-type]
        tables = payload["counts"]
        assert isinstance(tables, list)
        for n, flat in enumerate(tables):
            for ctx_str, bucket in flat.items():
                ctx = tuple(int(x) for x in ctx_str.split()) if ctx_str else ()
                counts = {int(t): int(c) for t, c in bucket.items()}
                model._counts[n][ctx] = counts
                model._totals[n][ctx] = sum(counts.values())
        return model

    def save(self, path: str) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str) -> NgramLM:
        return cls.from_dict(read_json(path))
