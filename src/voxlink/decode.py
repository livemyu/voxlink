"""解码策略（占位，实现见后续提交）。"""

from __future__ import annotations

from collections.abc import Sequence

from .llm import LanguageModel

__all__ = ["greedy_decode"]


def greedy_decode(  # pragma: no cover - 占位
    model: LanguageModel,
    prompt: Sequence[int],
    max_new_tokens: int,
    eos: int,
) -> list[int]:
    raise NotImplementedError
