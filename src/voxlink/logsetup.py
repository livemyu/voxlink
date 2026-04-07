"""统一的日志配置。

库代码只通过 ``get_logger`` 获取 logger，不主动配置 handler，把输出策略
留给应用层。:func:`configure` 是给 CLI / 示例用的便捷入口。
"""

from __future__ import annotations

import logging

_ROOT = "voxlink"


def get_logger(name: str | None = None) -> logging.Logger:
    """获取 voxlink 命名空间下的 logger。"""
    if name is None:
        return logging.getLogger(_ROOT)
    return logging.getLogger(f"{_ROOT}.{name}")


def configure(level: int = logging.INFO) -> None:
    """为 voxlink 顶层 logger 挂上一个简单的 stderr handler。"""
    logger = logging.getLogger(_ROOT)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
