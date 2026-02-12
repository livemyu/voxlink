"""确定性合成语音数据。

真实语音语料庞大且难以随仓库分发。为了让示例、测试与 CI 完全离线且可复现，
这里用简单的加性正弦 + 噪声，为一组「词」各自合成可区分的音频片段，并把它
们拼接成「话语」。同一段文本总是合成同样的波形，因此整条语音→单元→LLM→文本
的链路都是确定性的。
"""

from __future__ import annotations

import math

from ._random import Rng
from .types import Utterance, Vector, Waveform

__all__ = ["synth_word", "synth_utterance", "WORDS"]

# 一组玩具「词」，各自对应一个基频，方便前端区分。
WORDS: dict[str, float] = {
    "zero": 180.0,
    "one": 220.0,
    "two": 300.0,
    "three": 380.0,
    "four": 460.0,
    "five": 540.0,
    "yes": 620.0,
    "no": 700.0,
    "up": 780.0,
    "down": 860.0,
}


def _stable_seed(word: str) -> int:
    """与进程无关的确定性种子（避免 Python 内置 hash 的随机化）。"""
    acc = 0
    for ch in word:
        acc = (acc * 131 + ord(ch)) & 0xFFFFFF
    return acc


def synth_word(word: str, sample_rate: int = 16000, duration: float = 0.2) -> Vector:
    """为单个词合成一段带谐波与轻微噪声的正弦波形。"""
    base = WORDS.get(word, 200.0)
    rng = Rng(_stable_seed(word))
    n = int(sample_rate * duration)
    out: Vector = []
    for t in range(n):
        x = t / sample_rate
        val = (
            0.6 * math.sin(2 * math.pi * base * x)
            + 0.3 * math.sin(2 * math.pi * 2 * base * x)
            + 0.1 * math.sin(2 * math.pi * 3 * base * x)
        )
        val += 0.02 * rng.gauss()
        out.append(val)
    return out
