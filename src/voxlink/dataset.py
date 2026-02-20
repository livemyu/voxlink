"""数据清单（manifest）的加载与切分。

manifest 是 JSONL，每行形如 ``{"audio": "path.wav", "text": "..."}``。
:func:`load_manifest` 读入并（可选）加载音频，:func:`train_test_split` 提供
确定性划分。
"""

from __future__ import annotations

from ._random import Rng
from .audio import load_wav
from .io_utils import read_jsonl
from .types import Utterance, Waveform

__all__ = ["load_manifest", "train_test_split"]


def load_manifest(path: str, *, load_audio: bool = True) -> list[Utterance]:
    """从 JSONL manifest 读取话语列表。"""
    utts: list[Utterance] = []
    for rec in read_jsonl(path):
        text = str(rec.get("text", ""))
        if load_audio and "audio" in rec:
            audio = load_wav(str(rec["audio"]))
        else:
            audio = Waveform([], int(rec.get("sample_rate", 16000)))
        meta = {k: str(v) for k, v in rec.items() if k not in ("audio", "text")}
        utts.append(Utterance(audio=audio, text=text, meta=meta))
    return utts


def train_test_split(
    items: list[Utterance],
    *,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> tuple[list[Utterance], list[Utterance]]:
    """按比例确定性地划分训练 / 测试集。"""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio 需落在 (0, 1)")
    idx = list(range(len(items)))
    Rng(seed).shuffle(idx)
    n_test = max(1, int(len(items) * test_ratio)) if items else 0
    test_idx = set(idx[:n_test])
    train = [x for i, x in enumerate(items) if i not in test_idx]
    test = [x for i, x in enumerate(items) if i in test_idx]
    return train, test
