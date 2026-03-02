"""voxlink 命令行入口。

提供从数据合成、码本 / 语言模型训练，到转写、问答与评估的一站式子命令。
所有子命令都走纯 Python 内核，可离线运行。
"""

from __future__ import annotations

import argparse

from . import __version__
from .config import FrontendConfig, ModelConfig, UnitConfig
from .data import synth_utterance
from .eval import corpus_metrics
from .features import Frontend
from .io_utils import read_jsonl, write_json
from .logging_ import configure, get_logger
from .pipeline import SpeechLMPipeline

__all__ = ["build_parser", "main"]

_LOG = get_logger("cli")

# 一个便于演示的小语料（词序列）。
_DEMO_SENTENCES = [
    ["one", "two", "three"],
    ["yes", "no"],
    ["up", "down", "up"],
    ["four", "five"],
    ["zero", "one", "two"],
    ["no", "yes", "no"],
]


def _demo_pipeline() -> tuple[SpeechLMPipeline, list]:
    utts = [synth_utterance(s) for s in _DEMO_SENTENCES]
    pipe = SpeechLMPipeline(
        FrontendConfig(n_mels=40, n_mfcc=13),
        UnitConfig(n_units=24, seed=0),
        ModelConfig(order=4, smoothing=0.001),
    ).fit(utts)
    return pipe, utts


def _cmd_demo(args: argparse.Namespace) -> int:
    configure()
    pipe, utts = _demo_pipeline()
    refs = [u.text for u in utts]
    hyps = [pipe.transcribe(u.audio, max_new_tokens=8) for u in utts]
    for ref, hyp in zip(refs, hyps, strict=True):
        print(f"ref: {ref:24} -> hyp: {hyp}")
    metrics = corpus_metrics(refs, hyps)
    print(f"WER={metrics.wer:.3f}  CER={metrics.cer:.3f}  utts={metrics.num_utts}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    from .audio import load_wav

    frontend = Frontend(FrontendConfig(n_mfcc=args.n_mfcc))
    wave = load_wav(args.audio)
    feats = frontend(wave)
    print(f"frames={feats.num_frames} dims={feats.num_dims} frame_rate={feats.frame_rate}")
    if args.out:
        write_json(args.out, {"frames": feats.frames, "frame_rate": feats.frame_rate})
        _LOG.info("特征已写入 %s", args.out)
    return 0


def _cmd_transcribe(args: argparse.Namespace) -> int:
    from .audio import load_wav

    pipe = SpeechLMPipeline.from_pretrained(args.model)
    wave = load_wav(args.audio)
    print(pipe.transcribe(wave, max_new_tokens=args.max_new_tokens, beam_size=args.beam))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    pipe = SpeechLMPipeline.from_pretrained(args.model)
    from .audio import load_wav

    refs: list[str] = []
    hyps: list[str] = []
    for rec in read_jsonl(args.manifest):
        wave = load_wav(str(rec["audio"]))
        refs.append(str(rec.get("text", "")))
        hyps.append(pipe.transcribe(wave, max_new_tokens=args.max_new_tokens))
    metrics = corpus_metrics(refs, hyps)
    print(f"WER={metrics.wer:.4f}  CER={metrics.cer:.4f}  utts={metrics.num_utts}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造顶层参数解析器与子命令。"""
    parser = argparse.ArgumentParser(
        prog="voxlink",
        description="离线优先的语音大模型工具包",
    )
    parser.add_argument("--version", action="version", version=f"voxlink {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="用合成数据端到端演示 ASR")
    p_demo.set_defaults(handler=_cmd_demo)

    p_feat = sub.add_parser("features", help="从 WAV 提取特征")
    p_feat.add_argument("audio", help="输入 WAV 路径")
    p_feat.add_argument("--n-mfcc", type=int, default=13)
    p_feat.add_argument("--out", default="", help="可选：特征输出 JSON 路径")
    p_feat.set_defaults(handler=_cmd_features)

    p_tr = sub.add_parser("transcribe", help="用已训练模型转写 WAV")
    p_tr.add_argument("audio", help="输入 WAV 路径")
    p_tr.add_argument("--model", required=True, help="模型目录")
    p_tr.add_argument("--max-new-tokens", type=int, default=32)
    p_tr.add_argument("--beam", type=int, default=1, help="beam 宽度（1 为贪心）")
    p_tr.set_defaults(handler=_cmd_transcribe)

    p_ev = sub.add_parser("eval", help="在 manifest 上评估 WER/CER")
    p_ev.add_argument("manifest", help="JSONL manifest 路径")
    p_ev.add_argument("--model", required=True, help="模型目录")
    p_ev.add_argument("--max-new-tokens", type=int, default=32)
    p_ev.set_defaults(handler=_cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    result: int = handler(args)
    return result
