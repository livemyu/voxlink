"""voxlink 命令行入口。

提供从数据合成、码本 / 语言模型训练，到转写、问答与评估的一站式子命令。
所有子命令都走纯 Python 内核，可离线运行。
"""

from __future__ import annotations

import argparse

from . import __version__

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """构造顶层参数解析器与子命令。"""
    parser = argparse.ArgumentParser(
        prog="voxlink",
        description="离线优先的语音大模型工具包",
    )
    parser.add_argument("--version", action="version", version=f"voxlink {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # 各子命令的参数在后续提交里逐步补全
    sub.add_parser("demo", help="用合成数据端到端演示 ASR")
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
