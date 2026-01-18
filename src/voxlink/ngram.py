"""带回退的 n-gram 语言模型（参考实现）。

在离散 token 序列（语音单元 + 文本 token 的联合词表）上统计 n-gram 频次，
用 add-k 平滑估计条件概率，并在高阶上下文未见时逐级回退到低阶。它足够
轻量，可在测试中过拟合小语料，用来端到端验证「语音接入 LLM」的推理链路。
"""

from __future__ import annotations

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
