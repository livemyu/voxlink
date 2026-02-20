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


class PromptBuilder:
    """把语音单元与任务指令组织成语言模型的输入 token id 序列。

    约定的模板（token 字面量）：

        <bos> <task> u_i u_j ... [ 文本 token ... ] <eos>

    其中末尾的 ``<eos>`` 只作为「提示结束、答案开始」的分隔；生成阶段模型在
    它之后继续输出答案 token。
    """

    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def unit_token(self, unit: int) -> str:
        return f"u_{unit}"

    def build(
        self,
        units: list[int],
        task: TaskSpec,
        text: str = "",
    ) -> list[int]:
        """构造提示 token id 序列。"""
        tokens: list[str] = [SpecialTokens.BOS, task.token()]
        tokens.extend(self.unit_token(u) for u in units)
        if task.needs_text:
            if not text:
                raise ValueError(f"任务 {task.name} 需要文本输入")
            tokens.extend(word_tokenize(text))
        tokens.append(SpecialTokens.EOS)
        return self.vocab.encode(tokens)

    def decode_answer(self, ids: list[int]) -> str:
        """把答案 token id 还原为文本，跳过特殊符号与语音单元。"""
        words = []
        for i in ids:
            tok = self.vocab.token(i)
            if tok in SpecialTokens.ALL or tok.startswith("u_") or tok.startswith("<"):
                continue
            words.append(tok)
        return " ".join(words)
