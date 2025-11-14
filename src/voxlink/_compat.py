"""可选后端探测。

voxlink 的核心内核是纯 Python 的；当环境中安装了 numpy 或 torch 时，
部分算子可以切换到更快的实现。这里集中做一次探测，避免在热路径里反复
``import``。
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache


@lru_cache(maxsize=None)
def has_numpy() -> bool:
    """环境中是否可用 numpy。"""
    return importlib.util.find_spec("numpy") is not None


@lru_cache(maxsize=None)
def has_torch() -> bool:
    """环境中是否可用 torch。"""
    return importlib.util.find_spec("torch") is not None


def require(name: str) -> None:
    """确保某个可选后端存在，否则给出清晰的安装提示。"""
    ok = {"numpy": has_numpy, "torch": has_torch}.get(name)
    if ok is None:
        raise ValueError(f"未知后端: {name}")
    if not ok():
        raise ImportError(
            f"该功能需要可选后端 '{name}'，请先安装：pip install 'voxlink[{name}]'"
        )
