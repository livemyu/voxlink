"""WAV 音频读写。

只依赖标准库 :mod:`wave` 与 :mod:`struct`，支持 8/16/24/32 位 PCM，
读入后统一转成 ``[-1, 1]`` 的浮点单声道波形（多声道取平均）。
"""

from __future__ import annotations

import struct
import wave

from .exceptions import AudioError
from .types import Vector, Waveform

__all__ = ["float_to_pcm", "load_wav", "pcm_to_float", "save_wav"]

_INT_MAX = {8: 127, 16: 32767, 24: 8388607, 32: 2147483647}


def pcm_to_float(samples: list[int], bits: int) -> Vector:
    """把整数 PCM 采样归一化到 ``[-1, 1]``。"""
    if bits not in _INT_MAX:
        raise AudioError(f"不支持的位深: {bits}")
    peak = _INT_MAX[bits]
    return [max(-1.0, s / peak) for s in samples]


def float_to_pcm(samples: Vector, bits: int) -> list[int]:
    """把 ``[-1, 1]`` 的浮点采样量化回整数 PCM（带饱和裁剪）。"""
    if bits not in _INT_MAX:
        raise AudioError(f"不支持的位深: {bits}")
    peak = _INT_MAX[bits]
    out = []
    for s in samples:
        clipped = min(1.0, max(-1.0, s))
        out.append(int(round(clipped * peak)))
    return out


def _unpack_pcm(raw: bytes, bits: int, n: int) -> list[int]:
    """把原始字节解成有符号整数采样。"""
    if bits == 8:
        # WAV 的 8-bit PCM 是无符号，偏移 128。
        return [b - 128 for b in raw[:n]]
    if bits == 16:
        return list(struct.unpack(f"<{n}h", raw[: n * 2]))
    if bits == 32:
        return list(struct.unpack(f"<{n}i", raw[: n * 4]))
    if bits == 24:
        out = []
        for i in range(n):
            chunk = raw[i * 3 : i * 3 + 3]
            val = int.from_bytes(chunk, "little", signed=True)
            out.append(val)
        return out
    raise AudioError(f"不支持的位深: {bits}")


def load_wav(path: str) -> Waveform:
    """读取 WAV 文件，返回单声道浮点波形。

    多声道会按通道平均降为单声道。
    """
    try:
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except (wave.Error, EOFError, FileNotFoundError) as exc:
        raise AudioError(f"无法读取 WAV 文件 {path!r}: {exc}") from exc

    bits = sampwidth * 8
    total = n_frames * n_channels
    ints = _unpack_pcm(raw, bits, total)
    floats = pcm_to_float(ints, bits)

    if n_channels > 1:
        mono = [
            sum(floats[i : i + n_channels]) / n_channels
            for i in range(0, len(floats), n_channels)
        ]
    else:
        mono = floats
    return Waveform(samples=mono, sample_rate=sample_rate)


def _pack_pcm(samples: list[int], bits: int) -> bytes:
    """把有符号整数采样打包成小端字节。"""
    if bits == 8:
        return bytes((s + 128) & 0xFF for s in samples)
    if bits == 16:
        return struct.pack(f"<{len(samples)}h", *samples)
    if bits == 32:
        return struct.pack(f"<{len(samples)}i", *samples)
    if bits == 24:
        buf = bytearray()
        for s in samples:
            buf += int(s).to_bytes(3, "little", signed=True)
        return bytes(buf)
    raise AudioError(f"不支持的位深: {bits}")


def save_wav(path: str, wave_obj: Waveform, *, bits: int = 16) -> None:
    """把单声道浮点波形写成 PCM WAV 文件。"""
    ints = float_to_pcm(wave_obj.samples, bits)
    frames = _pack_pcm(ints, bits)
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(bits // 8)
            wf.setframerate(wave_obj.sample_rate)
            wf.writeframes(frames)
    except wave.Error as exc:
        raise AudioError(f"无法写入 WAV 文件 {path!r}: {exc}") from exc
