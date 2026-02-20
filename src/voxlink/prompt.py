"""提示词构造：把语音单元、任务指令与文本上下文拼装成 token 序列。

voxlink 采用「离散单元 + 任务标记」的模板：先放特殊 BOS，再放任务标记
（如 ``<asr>`` / ``<qa>``），随后是语音单元 token（``u_<id>``），最后视任务
附上文本形式的问题或指令。语言模型在此前缀之后自回归地生成答案 token。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .text import word_tokenize
from .vocab import SpecialTokens, Vocab

__all__ = ["TaskSpec", "PromptBuilder", "ASR", "QA", "INSTRUCT"]


@dataclass(frozen=True)
class TaskSpec:
    """一个任务的提示词规格。

    Attributes:
        name: 任务名，同时作为任务标记的字面量（形如 ``<asr>``）。
        needs_text: 该任务是否在语音单元之后追加文本（问题 / 指令）。
    """

    name: str
    needs_text: bool = False
    task_token: str = field(default="")

    def token(self) -> str:
        return self.task_token or f"<{self.name}>"


ASR = TaskSpec(name="asr", needs_text=False)
QA = TaskSpec(name="qa", needs_text=True)
INSTRUCT = TaskSpec(name="instruct", needs_text=True)
