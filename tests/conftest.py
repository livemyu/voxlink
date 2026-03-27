"""共享的 pytest 夹具。"""

from __future__ import annotations

import pytest

from voxlink._random import Rng


@pytest.fixture
def rng() -> Rng:
    """固定种子的确定性随机数生成器。"""
    return Rng(1234)


@pytest.fixture
def tiny_corpus() -> list[list[str]]:
    """一个供多处复用的小词序列语料。"""
    return [
        ["one", "two", "three"],
        ["yes", "no"],
        ["up", "down"],
        ["four", "five"],
    ]
