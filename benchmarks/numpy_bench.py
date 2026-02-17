"""
cthreading — NUMPY CPU-HEAVY BENCHMARK
========================================

stdlib threading.Thread vs cthreading @auto_thread on CPU-intensive NumPy
workloads.  Each worker gets its own data (seeded RNG) and performs heavy
BLAS / LAPACK / FFT operations that release the GIL internally.

Workloads:
  A. Dense matrix multiply      (N×N @ N×N)
  B. Full SVD decomposition     (N×N)
  C. 2-D FFT                    (N×N)
  D. Symmetric eigendecomp      (N×N)
  E. Cholesky + triangular solve(N×N)

Usage:
    uv run examples/numpy_bench.py              # run ALL workloads
    uv run examples/numpy_bench.py matmul       # single workload
    uv run examples/numpy_bench.py svd fft      # multiple workloads
"""

from __future__ import annotations

import os

# Force BLAS / OpenMP to use 1 thread per call so Python-level threading
# provides the parallelism.  Without this, BLAS spawns its own threads and
# the machine becomes oversubscribed (N_python_threads × N_blas_threads).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import statistics  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402
from bench_format import (  # noqa: E402
    WORKERS,
    Metrics,
    export_json,
    print_header,
    print_suite_summary,
    record,
    section,
)

from cthreading import parallel_starmap  # noqa: E402

# ===================================================================
# CONFIG
# ===================================================================

SUITE = "numpy"
REPS = 5
BASE_SEED = 42
WARMUP_REPS = 1

SIZES: dict[str, int] = {
    "matmul":   1024,
    "svd":       512,
    "fft":      2048,
    "eig":       768,
    "cholesky": 1024,
}

WORKLOAD_ORDER = ["matmul", "svd", "fft", "eig", "cholesky"]

WORKLOAD_LABELS: dict[str, str] = {
    "matmul":   f"Dense matrix multiply {SIZES['matmul']}x{SIZES['matmul']}",
    "svd":      f"Full SVD decomposition {SIZES['svd']}x{SIZES['svd']}",
    "fft":      f"2-D FFT {SIZES['fft']}x{SIZES['fft']}",
    "eig":      f"Symmetric eigendecomp {SIZES['eig']}x{SIZES['eig']}",
    "cholesky": f"Cholesky + tri-solve {SIZES['cholesky']}x{SIZES['cholesky']}",
}

# ===================================================================
# WORKLOAD FUNCTIONS
# ===================================================================

def workload_matmul(rng: np.random.Generator, n: int) -> float:
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    C = A @ B
    return float(np.abs(C).mean())


def workload_svd(rng: np.random.Generator, n: int) -> float:
    A = rng.standard_normal((n, n))
    _U, S, _Vt = np.linalg.svd(A, full_matrices=False)
    return float(S.sum())


def workload_fft(rng: np.random.Generator, n: int) -> float:
    A = rng.standard_normal((n, n))
    F = np.fft.fft2(A)
    return float(np.abs(F).mean())


def workload_eig(rng: np.random.Generator, n: int) -> float:
    A = rng.standard_normal((n, n))
    A = (A + A.T) * 0.5
    vals, _vecs = np.linalg.eigh(A)
    return float(np.abs(vals).sum())


def workload_cholesky(rng: np.random.Generator, n: int) -> float:
    A = rng.standard_normal((n, n))
    A = A @ A.T + n * np.eye(n)
    L = np.linalg.cholesky(A)
    b = rng.standard_normal(n)
    x = np.linalg.solve(L @ L.T, b)
    return float(np.abs(x).sum())


WORKLOAD_FNS: dict[str, Callable[[np.random.Generator, int], float]] = {
    "matmul":   workload_matmul,
    "svd":      workload_svd,
    "fft":      workload_fft,
    "eig":      workload_eig,
    "cholesky": workload_cholesky,
}

# ===================================================================
# HELPERS
# ===================================================================

def _detect_numpy_info() -> dict[str, str]:
    info: dict[str, str] = {"numpy": np.__version__}
    try:
        cfg = np.show_config(mode="dicts")  # type: ignore[call-arg]
        if isinstance(cfg, dict):
            blas = cfg.get("Build Dependencies", {}).get("blas", {})
            if isinstance(blas, dict):
                info["blas"] = blas.get("name", "unknown")
                info["blas_version"] = blas.get("version", "?")
    except Exception:
        info["blas"] = "unknown"
    return info


# ===================================================================
# WARMUP
# ===================================================================

def warmup(workload_fn: Callable[[np.random.Generator, int], float], size: int) -> None:
    rng = np.random.default_rng(0)
    for _ in range(WARMUP_REPS):
        workload_fn(rng, size)


# ===================================================================
# SEQUENTIAL BASELINE
# ===================================================================

def bench_sequential(
    workload_fn: Callable[[np.random.Generator, int], float],
    size: int,
    total_ops: int,
) -> float:
    rng = np.random.default_rng(BASE_SEED)
    t0 = time.perf_counter()
    for _ in range(total_ops):
        workload_fn(rng, size)
    return time.perf_counter() - t0


# ===================================================================
# BENCH: STDLIB THREADING
# ===================================================================

def bench_stdlib(
    workload_fn: Callable[[np.random.Generator, int], float],
    size: int,
) -> tuple[float, list[float]]:
    checksums: list[float] = [0.0] * WORKERS

    def worker(tid: int) -> None:
        rng = np.random.default_rng(BASE_SEED + tid)
        cs = 0.0
        for _ in range(REPS):
            cs += workload_fn(rng, size)
        checksums[tid] = cs

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0, list(checksums)


# ===================================================================
# BENCH: CTHREADING parallel_starmap
# ===================================================================

def bench_pmap(
    workload_fn: Callable[[np.random.Generator, int], float],
    size: int,
) -> tuple[float, list[float]]:
    def compute_task(tid: int, sz: int, n_reps: int) -> tuple[int, float]:
        """Each task does ALL reps for one worker — same granularity as stdlib."""
        rng = np.random.default_rng(BASE_SEED + tid)
        cs = 0.0
        for _ in range(n_reps):
            cs += workload_fn(rng, sz)
        return (tid, cs)

    items = [(tid, size, REPS) for tid in range(WORKERS)]

    t0 = time.perf_counter()
    results = parallel_starmap(compute_task, items, num_workers=WORKERS)
    elapsed = time.perf_counter() - t0

    checksums = [0.0] * WORKERS
    for tid, cs in results:
        checksums[tid] = cs
    return elapsed, list(checksums)


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    args = [a.lower() for a in sys.argv[1:]]
    for a in args:
        if a not in WORKLOAD_ORDER:
            print(f"Unknown workload: '{a}'")
            print(f"Available: {', '.join(WORKLOAD_ORDER)}")
            sys.exit(1)

    selected = args if args else list(WORKLOAD_ORDER)

    np_info = _detect_numpy_info()
    blas = np_info.get("blas", "unknown")
    bver = np_info.get("blas_version", "")
    if bver:
        blas += f" {bver}"
    print_header(
        "NUMPY CPU-HEAVY BENCHMARK",
        extra_lines=[
            f"NumPy        : {np_info.get('numpy', '?')}",
            f"BLAS backend : {blas}",
            f"BLAS threads : {os.environ.get('OMP_NUM_THREADS', '?')}  (pinned to 1)",
            f"Reps/worker  : {REPS}",
        ],
    )

    total_ops = WORKERS * REPS

    for wname in selected:
        wfn = WORKLOAD_FNS[wname]
        size = SIZES[wname]
        label = WORKLOAD_LABELS[wname]

        section(f"WARMUP: {wname} (N={size})")
        print(f"    Running {WARMUP_REPS} warm-up rep(s) \u2026", end=" ", flush=True)
        warmup(wfn, size)
        print("done.")

        section(f"SEQUENTIAL BASELINE: {wname}")
        print("    Running \u2026", end=" ", flush=True)
        seq_time = bench_sequential(wfn, size, total_ops)
        print(f"{seq_time:.4f}s")

        print("    [1/2] stdlib threading \u2026", end=" ", flush=True)
        std_elapsed, std_cs = bench_stdlib(wfn, size)
        print(f"{std_elapsed:.4f}s")

        print("    [2/2] parallel_starmap \u2026", end=" ", flush=True)
        ct_elapsed, ct_cs = bench_pmap(wfn, size)
        print(f"{ct_elapsed:.4f}s")

        # checksum validation
        s_mean = statistics.mean(std_cs) if std_cs else 0.0
        c_mean = statistics.mean(ct_cs) if ct_cs else 0.0
        tol = max(abs(s_mean) * 1e-6, 1e-6)
        cs_ok = abs(s_mean - c_mean) < tol

        std_m = Metrics(
            label="stdlib", ops=total_ops, elapsed=std_elapsed,
            correct=True, seq_elapsed=seq_time,
            final_value=f"{s_mean:.6e}",
            extra={"checksums_match": "YES" if cs_ok else "NO"},
        )
        ct_m = Metrics(
            label="cthreading", ops=total_ops, elapsed=ct_elapsed,
            correct=cs_ok, seq_elapsed=seq_time,
            final_value=f"{c_mean:.6e}",
            extra={"checksums_match": "YES" if cs_ok else "NO"},
        )

        record(SUITE, f"{wname} ({label})", std_m, ct_m, show_seq=True)

    print_suite_summary(SUITE)
    export_json(os.path.join(os.path.dirname(__file__), f"results_{SUITE}.json"))


if __name__ == "__main__":
    main()
