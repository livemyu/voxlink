"""轻量文本处理：归一化与分词。

评估与语言模型都需要一致的文本预处理。这里只做最小、可预测的操作，
避免引入 nltk / jieba 之类的重依赖——保持离线、确定性。
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize(text: str, *, lower: bool = True, strip_punct: bool = True) -> str:
    """统一大小写、Unicode 形式与空白。

    - 归一化到 NFKC，让全角/半角标点一致；
    - 可选去除标点；
    - 折叠连续空白。
    """
    text = unicodedata.normalize("NFKC", text)
    if lower:
        text = text.lower()
    if strip_punct:
        text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def word_tokenize(text: str) -> list[str]:
    """按空白切词（先做归一化）。"""
    norm = normalize(text)
    return norm.split() if norm else []


def char_tokenize(text: str) -> list[str]:
    """按字符切分（用于 CER 及中文场景），忽略空白。"""
    norm = normalize(text, strip_punct=True)
    return [ch for ch in norm if not ch.isspace()]
