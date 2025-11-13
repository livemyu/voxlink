"""JSON / JSONL 读写辅助。

数据清单（manifest）与模型工件都用 JSON 存储，以保证跨平台、可读、
可版本控制。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

StrPath = str | Path


def read_json(path: StrPath) -> Any:
    """读取单个 JSON 文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: StrPath, obj: Any, *, indent: int = 2) -> None:
    """写出 JSON（UTF-8，保留非 ASCII 字符）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        f.write("\n")


def read_jsonl(path: StrPath) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL，跳过空行。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: StrPath, records: Iterable[dict[str, Any]]) -> int:
    """写出 JSONL，返回写入的记录数。"""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
            count += 1
    return count
