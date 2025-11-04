"""确定性随机数工具。

为了保证离线、可复现，voxlink 内核不使用全局 ``random`` 状态，而是显式传递
一个基于种子的 :class:`Rng`。它实现了一个简单的 32 位 xorshift 生成器，
在所有平台、所有 Python 版本上都给出相同的序列。
"""

from __future__ import annotations

import math

_UINT32 = 0xFFFFFFFF


class Rng:
    """可复现的伪随机数生成器（xorshift32）。"""

    __slots__ = ("_state",)

    def __init__(self, seed: int = 0) -> None:
        # 种子为 0 会让 xorshift 退化，因此偏移到非零。
        self._state = (seed ^ 0x9E3779B9) & _UINT32 or 0x1234ABCD

    def _next_u32(self) -> int:
        x = self._state
        x ^= (x << 13) & _UINT32
        x ^= x >> 17
        x ^= (x << 5) & _UINT32
        self._state = x & _UINT32
        return self._state

    def random(self) -> float:
        """返回 ``[0, 1)`` 区间内的浮点数。"""
        return self._next_u32() / (_UINT32 + 1)

    def randint(self, low: int, high: int) -> int:
        """返回 ``[low, high)`` 内的整数。"""
        if high <= low:
            raise ValueError("high 必须大于 low")
        return low + int(self.random() * (high - low))

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Box–Muller 变换得到的正态样本。"""
        u1 = self.random() or 1e-12
        u2 = self.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def shuffle(self, items: list[int]) -> None:
        """就地 Fisher–Yates 洗牌。"""
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i + 1)
            items[i], items[j] = items[j], items[i]
