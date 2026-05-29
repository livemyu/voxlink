"""简单的性能基准：纯 Python FFT vs 逐规模耗时。

不依赖任何基准框架，直接用 time.perf_counter 粗略测量。用于观察 radix-2 FFT
相对朴素 DFT 的加速，以及不同帧数下前端的吞吐。

运行:
    python benchmarks/bench_fft.py
"""

from __future__ import annotations

import time

from voxlink._random import Rng
from voxlink.fft import dft, fft


def _signal(n: int) -> list[float]:
    rng = Rng(0)
    return [rng.gauss() for _ in range(n)]


def _time(fn, *args, repeat: int = 5) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    print(f"{'n':>6} {'DFT(ms)':>12} {'FFT(ms)':>12} {'加速':>8}")
    for n in (64, 128, 256, 512, 1024):
        sig = _signal(n)
        t_dft = _time(dft, sig)
        t_fft = _time(fft, sig)
        speedup = t_dft / t_fft if t_fft else float("inf")
        print(f"{n:>6} {t_dft * 1e3:>12.3f} {t_fft * 1e3:>12.3f} {speedup:>7.1f}x")


if __name__ == "__main__":
    main()
