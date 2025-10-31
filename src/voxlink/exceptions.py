"""voxlink 的异常层次。

所有对外抛出的异常都继承自 :class:`VoxlinkError`，方便调用方统一捕获。
"""

from __future__ import annotations


class VoxlinkError(Exception):
    """voxlink 所有异常的基类。"""


class AudioError(VoxlinkError):
    """音频读写或格式相关的错误。"""


class FeatureError(VoxlinkError):
    """特征前端计算过程中的错误。"""


class ConfigError(VoxlinkError):
    """配置参数非法时抛出。"""


class ModelError(VoxlinkError):
    """模型（码本 / 语言模型）加载或使用错误。"""
