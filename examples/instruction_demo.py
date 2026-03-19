"""语音到文本的指令跟随示例。

演示 voxlink 的指令模板：语音单元之后拼接一条文本指令，模型据此生成文本
输出。这里的「指令」是玩具级别的（如 repeat / uppercase 的标签），旨在展示
接口而非语义能力——真实系统可在相同接口下换用更强的语言模型。

运行:
    python examples/instruction_demo.py
"""

from __future__ import annotations

from voxlink import FrontendConfig, ModelConfig, SpeechLMPipeline, UnitConfig
from voxlink.data import synth_utterance
from voxlink.prompt import INSTRUCT
from voxlink.types import Utterance

SAMPLES = [
    (["one", "two", "three"], "repeat", "one two three"),
    (["yes", "no"], "repeat", "yes no"),
    (["four", "five"], "repeat", "four five"),
]


def main() -> None:
    utts = [
        Utterance(audio=synth_utterance(words).audio, text=target, meta={"prompt": instr})
        for words, instr, target in SAMPLES
    ]

    pipe = SpeechLMPipeline(
        FrontendConfig(n_mels=40, n_mfcc=13),
        UnitConfig(n_units=20, seed=0),
        ModelConfig(order=5, smoothing=0.001),
    ).fit(utts, task=INSTRUCT)

    print("=== 指令跟随 ===")
    for (words, instr, _target), u in zip(SAMPLES, utts, strict=True):
        out = pipe.follow(u.audio, instr, max_new_tokens=8)
        print(f"  指令={instr:10} 语音={'/'.join(words):18} -> {out!r}")


if __name__ == "__main__":
    main()
