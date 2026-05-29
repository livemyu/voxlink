"""评估指标：词错误率 (WER) 与字错误率 (CER)。

两者都基于编辑距离（Levenshtein），对齐前先做文本归一化。WER 在词级、
CER 在字符级计算，后者更适合中文等无显式分词的语言。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .text import char_tokenize, word_tokenize

__all__ = ["CorpusMetrics", "cer", "corpus_metrics", "levenshtein", "wer"]


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


@dataclass(frozen=True)
class CorpusMetrics:
    """整个语料上的聚合指标。"""

    wer: float
    cer: float
    num_utts: int
    ref_words: int
    ref_chars: int

    def format(self) -> str:
        """给出一行可读的指标摘要。"""
        return (
            f"WER={self.wer:.4f} CER={self.cer:.4f} "
            f"utts={self.num_utts} words={self.ref_words} chars={self.ref_chars}"
        )


def corpus_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> CorpusMetrics:
    """在语料级聚合 WER/CER（按总编辑数 / 总长度，而非样本平均）。"""
    if len(references) != len(hypotheses):
        raise ValueError("参考与假设数量不一致")
    tot_word_err = tot_words = 0
    tot_char_err = tot_chars = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        rw, hw = word_tokenize(ref), word_tokenize(hyp)
        rc, hc = char_tokenize(ref), char_tokenize(hyp)
        tot_word_err += levenshtein(rw, hw)
        tot_words += len(rw)
        tot_char_err += levenshtein(rc, hc)
        tot_chars += len(rc)
    return CorpusMetrics(
        wer=tot_word_err / tot_words if tot_words else 0.0,
        cer=tot_char_err / tot_chars if tot_chars else 0.0,
        num_utts=len(references),
        ref_words=tot_words,
        ref_chars=tot_chars,
    )
