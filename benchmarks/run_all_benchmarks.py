"""
cthreading — MASTER BENCHMARK RUNNER
======================================

Runs all benchmark suites in sequence and prints a combined summary.

Benchmark suites:
  1. numpy_bench.py     — CPU-heavy NumPy BLAS/LAPACK/FFT
  2. pillow_bench.py    — Pillow image-processing (blur, resize, filters)
  3. pure_cpu_bench.py  — Pure Python CPU work (Monte Carlo, Mandelbrot, etc.)
  4. simulated_bench.py — Simulated CPU work (Monte Carlo, Mandelbrot, etc.)
  5. head_to_head.py    — Head-to-head comparison of cthreading vs stdlib threading

Usage:
    uv run examples/run_all_benchmarks.py           # run everything
    uv run examples/run_all_benchmarks.py numpy     # single suite
    uv run examples/run_all_benchmarks.py pillow cpu # multiple suites
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import typing

# bench_format lives in the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bench_format import (  # noqa: E402
    ALL_RESULTS,
    print_header, print_final_aggregate, load_json,
)

RESULTS_FILE = "all_bench_results.txt"


class _Tee(io.TextIOBase):
    """Write to both the original stdout and a log file."""

    def __init__(self, file: io.TextIOWrapper, original: typing.TextIO) -> None:
        self._file = file
        self._original = original

    def write(self, s: str) -> int:  # type: ignore[override]
        self._original.write(s)
        self._file.write(s)
        return len(s)

    def flush(self) -> None:
        self._original.flush()
        self._file.flush()


SUITES: dict[str, dict[str, str]] = {
    "numpy": {
        "script": "numpy_bench.py",
        "label": "NumPy CPU-Heavy (BLAS/LAPACK/FFT)",
        "json": "results_numpy.json",
    },
    "pillow": {
        "script": "pillow_bench.py",
        "label": "Pillow Image Processing",
        "json": "results_pillow.json",
    },
    "cpu": {
        "script": "pure_cpu_bench.py",
        "label": "Pure Python CPU (free-threaded showcase)",
        "json": "results_cpu.json",
    },
    "simulated": {
        "script": "simulated_bench.py",
        "label": "Simulated system benchmarks",
        "json": "results_simulated.json",
    },
    "head_to_head": {
        "script": "head_to_head.py",
        "label": "Head-to-head (auto_threaded edition)",
        "json": "results_head_to_head.json",
    },
}

SUITE_ORDER = ["numpy", "pillow", "cpu", "simulated", "head_to_head"]


def run_suite(key: str, bench_dir: str) -> tuple[str, float, int]:
    """Run a benchmark suite as a subprocess. Returns (key, elapsed, returncode)."""
    info = SUITES[key]
    script_path = os.path.join(bench_dir, info["script"])

    if not os.path.exists(script_path):
        print(f"  [ERROR] Script not found: {script_path}")
        return key, 0.0, 1

    from bench_format import section
    section(f"RUNNING: {info['label']}  ({info['script']})")
    print()

    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=os.path.dirname(bench_dir),  # project root
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = time.perf_counter() - t0

    if result.stdout:
        print(result.stdout, end="")

    print()
    if result.returncode == 0:
        print(f"  [{key}] completed in {elapsed:.1f}s")
    else:
        print(f"  [{key}] FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
    print()

    return key, elapsed, result.returncode


def main() -> None:
    args = [a.lower() for a in sys.argv[1:]]

    for a in args:
        if a not in SUITES:
            print(f"Unknown suite: '{a}'")
            print(f"Available: {', '.join(SUITE_ORDER)}")
            sys.exit(1)

    selected = args if args else list(SUITE_ORDER)

    bench_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(bench_dir)
    results_path = os.path.join(project_root, RESULTS_FILE)
    log_file = open(results_path, "w", encoding="utf-8")
    original_stdout = sys.__stdout__ or sys.stdout
    sys.stdout = _Tee(log_file, original_stdout)

    suite_list = ", ".join(selected)
    print_header(
        "MASTER BENCHMARK RUNNER",
        extra_lines=[f"Suites       : {suite_list}"],
    )

    run_results: list[tuple[str, float, int]] = []
    master_t0 = time.perf_counter()

    for key in selected:
        result = run_suite(key, bench_dir)
        run_results.append(result)

    master_elapsed = time.perf_counter() - master_t0

    # ── Load JSON from each suite and build the aggregate ──
    for key in selected:
        json_path = os.path.join(bench_dir, SUITES[key]["json"])
        if os.path.exists(json_path):
            loaded = load_json(json_path)
            ALL_RESULTS.extend(loaded)

    print_final_aggregate()

    # ── Execution summary ──
    from bench_format import section as _sec
    _sec(f"COMPLETE \u2014 total wall time: {master_elapsed:.1f}s")
    for key, elapsed, rc in run_results:
        status = "\033[32mOK\033[0m" if rc == 0 else "\033[31mFAILED\033[0m"
        print(f"    {key:<14s}  {elapsed:>7.1f}s  {status}")

    sys.stdout = sys.__stdout__  # type: ignore[assignment]
    log_file.close()
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
