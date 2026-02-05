"""评估指标：词错误率 (WER) 与字错误率 (CER)。

两者都基于编辑距离（Levenshtein），对齐前先做文本归一化。WER 在词级、
CER 在字符级计算，后者更适合中文等无显式分词的语言。
"""

from __future__ import annotations

from collections.abc import Sequence

from .text import char_tokenize, word_tokenize

__all__ = ["levenshtein", "wer", "cer"]


def levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    """两个序列之间的编辑距离（插入/删除/替换各计 1）。"""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,  # 删除
                curr[j - 1] + 1,  # 插入
                prev[j - 1] + cost,  # 替换 / 匹配
            )
        prev = curr
    return prev[m]


def wer(reference: str, hypothesis: str) -> float:
    """词错误率：词级编辑距离 / 参考词数。

    参考为空时，假设也为空则 WER=0，否则 WER=1。
    """
    ref_words = word_tokenize(reference)
    hyp_words = word_tokenize(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    """字错误率：字符级编辑距离 / 参考字符数。"""
    ref_chars = char_tokenize(reference)
    hyp_chars = char_tokenize(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein(ref_chars, hyp_chars) / len(ref_chars)
