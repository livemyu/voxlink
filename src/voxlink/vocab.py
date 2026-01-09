"""联合词表：把离散语音单元与文本 token 放进同一个 id 空间。

语音接入 LLM 的常见做法是扩展词表：前一段 id 留给语音单元，后一段留给
文本 token，另外保留若干特殊符号（BOS/EOS/PAD 及任务标记）。这个模块提供
一个最小、可序列化的词表实现。
"""

from __future__ import annotations

from .exceptions import ModelError
from .io_utils import read_json, write_json

__all__ = ["Vocab", "SpecialTokens"]


class SpecialTokens:
    """特殊符号的字面量集合。"""

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    ALL = (PAD, BOS, EOS, UNK)


class Vocab:
    """token <-> id 的双向映射。"""

    def __init__(self, tokens: list[str]) -> None:
        if len(set(tokens)) != len(tokens):
            raise ModelError("词表中存在重复 token")
        self._itos = list(tokens)
        self._stoi = {t: i for i, t in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self._itos)

    def __contains__(self, token: str) -> bool:
        return token in self._stoi

    def id(self, token: str) -> int:
        """token 转 id；未登录时回退到 ``<unk>``。"""
        if token in self._stoi:
            return self._stoi[token]
        return self._stoi[SpecialTokens.UNK]

    def token(self, idx: int) -> str:
        """id 转 token。"""
        if not 0 <= idx < len(self._itos):
            raise ModelError(f"id 越界: {idx}")
        return self._itos[idx]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.id(t) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.token(i) for i in ids]

    def save(self, path: str) -> None:
        write_json(path, {"tokens": self._itos})

    @classmethod
    def load(cls, path: str) -> Vocab:
        payload = read_json(path)
        return cls(payload["tokens"])
