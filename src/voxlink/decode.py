"""自回归解码策略。

在 :class:`~voxlink.llm.LanguageModel` 之上实现贪心、采样与 beam search。
所有策略都接受一个 ``eos`` token 与 ``max_new_tokens`` 上限。
"""

from __future__ import annotations

from collections.abc import Sequence

from ._random import Rng
from .llm import LanguageModel

__all__ = ["greedy_decode", "sample_decode", "beam_search"]


def _argmax(values: Sequence[float]) -> int:
    best = 0
    for i in range(1, len(values)):
        if values[i] > values[best]:
            best = i
    return best


def greedy_decode(
    model: LanguageModel,
    prompt: Sequence[int],
    max_new_tokens: int,
    eos: int,
) -> list[int]:
    """每步取概率最高的 token，直到 EOS 或达到上限。"""
    generated: list[int] = []
    context = list(prompt)
    for _ in range(max_new_tokens):
        token = _argmax(model.logprobs(context))
        if token == eos:
            break
        generated.append(token)
        context.append(token)
    return generated
