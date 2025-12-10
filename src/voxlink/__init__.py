"""voxlink: 离线优先的语音大模型工具包。

把语音编码器 / 离散语音 token 接入语言模型，实现语音理解、语音问答与
语音到文本的指令跟随。核心内核为纯 Python 实现。
"""

from __future__ import annotations

from .config import FrontendConfig, ModelConfig, UnitConfig
from .exceptions import (
    AudioError,
    ConfigError,
    FeatureError,
    ModelError,
    VoxlinkError,
)
from .types import FeatureMatrix, Utterance, Waveform

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "FrontendConfig",
    "UnitConfig",
    "ModelConfig",
    "Waveform",
    "FeatureMatrix",
    "Utterance",
    "VoxlinkError",
    "AudioError",
    "FeatureError",
    "ConfigError",
    "ModelError",
]
