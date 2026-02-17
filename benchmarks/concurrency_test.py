"""
This concurrency test is actively being used to test the cthreading auto_threaded decorator/function wrapper.
auto_threaded is currently broken and under active repair.

The goal with the bench is to see both normal times against sequential in true sequential tasks automatically,
while also seeing the speedup of parallelism in true parallel tasks automatically without having to manually
write (c)threading code.
"""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from cthreading import auto_threaded

if TYPE_CHECKING:
    from collections.abc import Callable

# ============================================================
# CONFIG
# ============================================================

TOTAL_WORK: int = 3_000_000
MAX_THREADS: int = os.cpu_count() or 8
TEST_THREADS: list[int] = [1, 2, 4, MAX_THREADS]

random.seed(0)


# ============================================================
# WORKLOAD 1: Branch Heavy / GIL Heavy
# ============================================================

def chaotic_work(n: int) -> int:
    total = 0
    for i in range(n):
        x = i ^ (i >> 3)

        if x % 7 == 0:
            total += x * 3
        elif x % 5 == 0:
            total += x // 2
        elif x % 3 == 0:
            total += x * x
        else:
            total += x

        if random.random() < 0.0005:
            for _ in range(50):
                total += _ * x

    return total


# ============================================================
# WORKLOAD 2: Uneven / Straggler Test
# ============================================================

def uneven_work(n: int) -> int:
    total = 0
    for i in range(n):
        total += i

        if i % 10000 == 0:
            for _ in range(20000):
                total += i ^ _
    return total


# ============================================================
# SEQUENTIAL RUNNER
# ============================================================

def run_sequential(func: Callable[[int], int | float]) -> float:
    start = time.perf_counter()
    func(TOTAL_WORK)
    return time.perf_counter() - start


# ============================================================
# THREADPOOL EXECUTOR RUNNER
# ============================================================

def run_threadpool(func: Callable[[int], int | float], workers: int) -> float:
    chunk = TOTAL_WORK // workers

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures: list[Future[int | float]] = [ex.submit(func, chunk) for _ in range(workers)]
        [f.result(timeout=1e6) for f in futures]

    return time.perf_counter() - start


# ============================================================
# CTHREADING RUNNER
# ============================================================

def run_cthreading(func: Callable[[int], int | float]) -> float:
    start = time.perf_counter()
    try:
        wrapped = auto_threaded(func)
        wrapped(TOTAL_WORK)
    finally:
        if auto_threaded.is_patched():
            auto_threaded.unpatch()
    return time.perf_counter() - start


# ============================================================
# BENCHMARK DRIVER
# ============================================================

def benchmark(label: str, func: Callable[[int], int | float], *, skip_threadpool: bool = False) -> None:
    print(f"\n===== {label} =====")

    seq_time = run_sequential(func)
    print(f"Sequential: {seq_time:.4f}s")

    for t in TEST_THREADS:
        if not skip_threadpool:
            tp_time = run_threadpool(func, t)
            speedup = seq_time / tp_time
            print(f"ThreadPool ({t} threads): {tp_time:.4f}s | Speedup: {speedup:.2f}x")

    ct_time = run_cthreading(func)
    speedup = seq_time / ct_time
    print(f"cthreading (adaptive): {ct_time:.4f}s | Speedup: {speedup:.2f}x")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(f"CPU Threads Available: {MAX_THREADS}")

    benchmark("Chaotic Workload", chaotic_work)
    benchmark("Uneven Workload", uneven_work)


if __name__ == "__main__":
    main()

    print("\n===== auto_threaded (patched stdlib) =====")
    def _auto_main() -> None:
        benchmark("Chaotic Workload", chaotic_work, skip_threadpool=True)
        benchmark("Uneven Workload", uneven_work, skip_threadpool=True)

    auto_threaded.run(_auto_main)
