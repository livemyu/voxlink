"""端到端语音转写（ASR）示例。

用内置的确定性合成语料训练一个 voxlink 管线，然后转写其中几条音频，
并打印 WER / CER。全程离线，无需任何外部数据或模型下载。

运行:
    python examples/asr_demo.py
"""

from __future__ import annotations

from voxlink import FrontendConfig, ModelConfig, SpeechLMPipeline, UnitConfig
from voxlink.data import synth_utterance
from voxlink.eval import corpus_metrics

SENTENCES = [
    ["one", "two", "three"],
    ["yes", "no"],
    ["up", "down", "up"],
    ["four", "five", "zero"],
    ["no", "yes"],
]


def main() -> None:
    utts = [synth_utterance(words) for words in SENTENCES]

    pipe = SpeechLMPipeline(
        FrontendConfig(n_mels=40, n_mfcc=13),
        UnitConfig(n_units=24, seed=0),
        ModelConfig(order=4, smoothing=0.001),
    ).fit(utts)

    refs = [u.text for u in utts]
    hyps = [pipe.transcribe(u.audio, max_new_tokens=8) for u in utts]

    print("=== 语音转写结果 ===")
    for ref, hyp in zip(refs, hyps, strict=True):
        flag = "✓" if ref == hyp else "✗"
        print(f"  {flag} 参考: {ref:22} 输出: {hyp}")

    m = corpus_metrics(refs, hyps)
    print(f"\nWER = {m.wer:.3f}   CER = {m.cer:.3f}   句数 = {m.num_utts}")


if __name__ == "__main__":
    main()
