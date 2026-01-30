"""自回归解码策略。

在 :class:`~voxlink.llm.LanguageModel` 之上实现贪心、采样与 beam search。
所有策略都接受一个 ``eos`` token 与 ``max_new_tokens`` 上限。
"""

from __future__ import annotations

from collections.abc import Sequence

from ._random import Rng
from .llm import LanguageModel
from .mathx import softmax

__all__ = ["greedy_decode", "sample_decode", "beam_search", "top_k_filter"]


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


def top_k_filter(logprobs: Sequence[float], k: int) -> list[float]:
    """只保留 top-k 的对数概率，其余置为 ``-inf``。"""
    if k <= 0 or k >= len(logprobs):
        return list(logprobs)
    threshold = sorted(logprobs, reverse=True)[k - 1]
    return [lp if lp >= threshold else float("-inf") for lp in logprobs]


def sample_decode(
    model: LanguageModel,
    prompt: Sequence[int],
    max_new_tokens: int,
    eos: int,
    *,
    rng: Rng,
    temperature: float = 1.0,
    top_k: int = 0,
) -> list[int]:
    """带温度与可选 top-k 的采样解码。"""
    if temperature <= 0:
        raise ValueError("temperature 必须为正")
    generated: list[int] = []
    context = list(prompt)
    for _ in range(max_new_tokens):
        logp = [lp / temperature for lp in model.logprobs(context)]
        if top_k:
            logp = top_k_filter(logp, top_k)
        probs = softmax(logp)
        token = _sample_from(probs, rng)
        if token == eos:
            break
        generated.append(token)
        context.append(token)
    return generated


def _sample_from(probs: Sequence[float], rng: Rng) -> int:
    target = rng.random()
    acc = 0.0
    for token, p in enumerate(probs):
        acc += p
        if acc >= target:
            return token
    return len(probs) - 1
