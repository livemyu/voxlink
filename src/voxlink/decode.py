"""自回归解码策略。

在 :class:`~voxlink.llm.LanguageModel` 之上实现贪心、采样与 beam search。
所有策略都接受一个 ``eos`` token 与 ``max_new_tokens`` 上限。
"""

from __future__ import annotations

from collections.abc import Sequence

from ._random import Rng
from .llm import LanguageModel
from .mathx import softmax

__all__ = [
    "greedy_decode",
    "sample_decode",
    "beam_search",
    "top_k_filter",
    "StopCriteria",
]


class StopCriteria:
    """解码停止条件：命中任一 stop token 或达到最大长度。"""

    def __init__(self, eos: int, max_new_tokens: int, stop_tokens: Sequence[int] = ()) -> None:
        self.max_new_tokens = max_new_tokens
        self._stops = {eos, *stop_tokens}

    def is_stop_token(self, token: int) -> bool:
        return token in self._stops

    def reached_limit(self, n_generated: int) -> bool:
        return n_generated >= self.max_new_tokens


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


def beam_search(
    model: LanguageModel,
    prompt: Sequence[int],
    max_new_tokens: int,
    eos: int,
    *,
    beam_size: int = 4,
    length_penalty: float = 0.0,
) -> list[int]:
    """定宽 beam search，返回得分最高的候选序列。

    得分为累计对数概率，可选长度惩罚 ``score / len**alpha`` 以缓解 beam
    search 偏好短序列的问题。
    """
    if beam_size < 1:
        raise ValueError("beam_size 必须为正")
    # 每个 beam: (tokens, score, finished)
    beams: list[tuple[list[int], float, bool]] = [([], 0.0, False)]
    for _ in range(max_new_tokens):
        if all(finished for _, _, finished in beams):
            break
        candidates: list[tuple[list[int], float, bool]] = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, True))
                continue
            logp = model.logprobs(list(prompt) + tokens)
            ranked = sorted(range(len(logp)), key=lambda t: logp[t], reverse=True)
            for token in ranked[:beam_size]:
                new_tokens = [*tokens, token]
                new_score = score + logp[token]
                candidates.append((new_tokens, new_score, token == eos))
        candidates.sort(key=lambda b: _beam_priority(b, length_penalty), reverse=True)
        beams = candidates[:beam_size]
    best = max(beams, key=lambda b: _beam_priority(b, length_penalty))
    tokens = best[0]
    if tokens and tokens[-1] == eos:
        tokens = tokens[:-1]
    return tokens


def _beam_priority(beam: tuple[list[int], float, bool], length_penalty: float) -> float:
    tokens, score, _ = beam
    if length_penalty and tokens:
        return score / (len(tokens) ** length_penalty)
    return score
