"""
cthreading — PURE PYTHON CPU-HEAVY BENCHMARK
===============================================

stdlib threading.Thread vs cthreading @auto_thread on pure-Python CPU work.
NO C extensions (numpy, etc.) — every cycle is Python bytecode, which means
free-threaded Python 3.14t can truly parallelise these across cores.

This is THE showcase for cthreading + free-threaded Python.

Workloads:
  A. Monte Carlo Pi     — random sampling, tight arithmetic loop
  B. Mandelbrot set     — complex-number iteration per pixel
  C. Prime sieve        — trial division over large ranges
  D. N-body gravity     — O(N²) force computation, pure math
  E. JSON round-trip    — serialize + deserialize large dicts

Usage:
    uv run examples/pure_cpu_bench.py                # run ALL
    uv run examples/pure_cpu_bench.py mandelbrot     # single workload
    uv run examples/pure_cpu_bench.py primes nbody   # multiple
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from typing import TYPE_CHECKING

from bench_format import (
    CPUS,
    WORKERS,
    Metrics,
    export_json,
    print_header,
    print_suite_summary,
    record,
    section,
)

from cthreading import parallel_starmap

if TYPE_CHECKING:
    from collections.abc import Callable

# ===================================================================
# CONFIG
# ===================================================================

SUITE = "cpu"
BASE_SEED = 12345

# Workload sizes — tuned for ~1-4 s sequential per chunk on a modern CPU.
MONTE_CARLO_SAMPLES = 5_000_000
MANDELBROT_SIZE = 1400         # 1400×1400 grid
PRIME_LIMIT = 1_500_000
NBODY_PARTICLES = 512
NBODY_STEPS = 40
JSON_DOCS = 300
JSON_KEYS = 500

WORKLOAD_ORDER = ["montecarlo", "mandelbrot", "primes", "nbody", "json"]

# ===================================================================
# WORKLOAD FUNCTIONS
# ===================================================================
# Each takes (seed, chunk_id) and returns a numeric result for verification.

def workload_montecarlo(seed: int, chunk_id: int) -> float:
    """Estimate Pi via Monte Carlo sampling — tight arithmetic loop."""
    rng = random.Random(seed + chunk_id)
    inside = 0
    n = MONTE_CARLO_SAMPLES
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return 4.0 * inside / n


MANDELBROT_BANDS = CPUS * 6  # fine-grained for work-stealing

def workload_mandelbrot(seed: int, chunk_id: int) -> float:
    """Compute a band of the Mandelbrot set — complex iteration per pixel."""
    size = MANDELBROT_SIZE
    max_iter = 100
    bands = MANDELBROT_BANDS
    band_height = size // bands
    y_start = chunk_id * band_height
    y_end = y_start + band_height if chunk_id < bands - 1 else size

    total_iters = 0
    for py in range(y_start, y_end):
        y0 = (py - size / 2) * 4.0 / size
        for px in range(size):
            x0 = (px - size / 2) * 4.0 / size
            x, y = 0.0, 0.0
            iteration = 0
            while x * x + y * y <= 4.0 and iteration < max_iter:
                xtemp = x * x - y * y + x0
                y = 2.0 * x * y + y0
                x = xtemp
                iteration += 1
            total_iters += iteration
    return float(total_iters)


PRIME_CHUNKS = CPUS * 6  # fine-grained for work-stealing

def workload_primes(seed: int, chunk_id: int) -> float:
    """Count primes in a range using trial division."""
    chunk_size = PRIME_LIMIT // PRIME_CHUNKS
    start = max(2, chunk_id * chunk_size)
    end = (chunk_id + 1) * chunk_size if chunk_id < PRIME_CHUNKS - 1 else PRIME_LIMIT

    count = 0
    for n in range(start, end):
        if n < 2:
            continue
        is_prime = True
        limit = int(math.isqrt(n)) + 1
        for d in range(2, limit):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return float(count)


def workload_nbody(seed: int, chunk_id: int) -> float:
    """N-body gravitational simulation — O(N²) force computation."""
    rng = random.Random(seed + chunk_id)
    n = NBODY_PARTICLES
    steps = NBODY_STEPS
    dt = 0.01
    G = 1.0

    # Initialise particles: [x, y, vx, vy, mass]
    particles: list[list[float]] = []
    for _ in range(n):
        particles.append([
            rng.uniform(-10, 10),   # x
            rng.uniform(-10, 10),   # y
            rng.uniform(-1, 1),     # vx
            rng.uniform(-1, 1),     # vy
            rng.uniform(0.5, 5.0),  # mass
        ])

    for _ in range(steps):
        # Compute forces
        forces = [[0.0, 0.0] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dx = particles[j][0] - particles[i][0]
                dy = particles[j][1] - particles[i][1]
                dist_sq = dx * dx + dy * dy + 0.01  # softening
                inv_dist = 1.0 / math.sqrt(dist_sq)
                f = G * particles[i][4] * particles[j][4] * inv_dist * inv_dist * inv_dist
                fx = f * dx
                fy = f * dy
                forces[i][0] += fx
                forces[i][1] += fy
                forces[j][0] -= fx
                forces[j][1] -= fy
        # Integrate
        for i in range(n):
            inv_m = 1.0 / particles[i][4]
            particles[i][2] += forces[i][0] * inv_m * dt
            particles[i][3] += forces[i][1] * inv_m * dt
            particles[i][0] += particles[i][2] * dt
            particles[i][1] += particles[i][3] * dt

    # Return total kinetic energy as checksum
    ke = 0.0
    for p in particles:
        ke += 0.5 * p[4] * (p[2] * p[2] + p[3] * p[3])
    return ke


def workload_json(seed: int, chunk_id: int) -> float:
    """JSON round-trip: build, serialize, deserialize, hash — pure Python I/O."""
    rng = random.Random(seed + chunk_id)
    total_bytes = 0

    for doc_i in range(JSON_DOCS):
        doc = {}
        for k in range(JSON_KEYS):
            key = f"field_{doc_i}_{k}"
            doc[key] = {
                "value": rng.random(),
                "label": hashlib.md5(key.encode()).hexdigest(),
                "tags": [rng.randint(0, 1000) for _ in range(5)],
                "nested": {"a": rng.random(), "b": rng.gauss(0, 1)},
            }
        blob = json.dumps(doc)
        total_bytes += len(blob)
        parsed = json.loads(blob)
        total_bytes += len(parsed)

    return float(total_bytes)


WORKLOAD_FNS: dict[str, Callable[[int, int], float]] = {
    "montecarlo": workload_montecarlo,
    "mandelbrot": workload_mandelbrot,
    "primes":     workload_primes,
    "nbody":      workload_nbody,
    "json":       workload_json,
}

WORKLOAD_LABELS: dict[str, str] = {
    "montecarlo": f"Monte Carlo Pi ({MONTE_CARLO_SAMPLES:,} samples/worker)",
    "mandelbrot": f"Mandelbrot {MANDELBROT_SIZE}x{MANDELBROT_SIZE} ({MANDELBROT_BANDS} bands)",
    "primes":     f"Prime sieve to {PRIME_LIMIT:,} ({PRIME_CHUNKS} chunks)",
    "nbody":      f"N-body {NBODY_PARTICLES} particles x {NBODY_STEPS} steps",
    "json":       f"JSON round-trip {JSON_DOCS} docs x {JSON_KEYS} keys",
}

# Number of items to dispatch for each workload when using parallel_starmap
# (more items than workers = work-stealing load balance)
WORKLOAD_ITEMS: dict[str, int] = {
    "montecarlo": WORKERS,
    "mandelbrot": MANDELBROT_BANDS,
    "primes":     PRIME_CHUNKS,
    "nbody":      WORKERS,
    "json":       WORKERS,
}


# ===================================================================
# SEQUENTIAL BASELINE
# ===================================================================

def bench_sequential(
    workload_fn: Callable[[int, int], float],
    num_workers: int,
) -> tuple[float, list[float]]:
    """Run all chunks sequentially. Returns (wall_time, checksums)."""
    checksums: list[float] = []
    t0 = time.perf_counter()
    for i in range(num_workers):
        cs = workload_fn(BASE_SEED, i)
        checksums.append(cs)
    wall = time.perf_counter() - t0
    return wall, checksums


# ===================================================================
# BENCH: STDLIB THREADING
# ===================================================================

def bench_stdlib(
    workload_fn: Callable[[int, int], float],
    num_items: int,
) -> tuple[float, list[float]]:
    checksums: list[float] = [0.0] * num_items

    def worker(tid: int) -> None:
        for i in range(tid, num_items, WORKERS):
            checksums[i] = workload_fn(BASE_SEED, i)

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
    workload_fn: Callable[[int, int], float],
    num_items: int,
) -> tuple[float, list[float]]:
    def coarse_worker(tid: int, n_items: int, n_workers: int) -> list[tuple[int, float]]:
        """Each task handles its share of chunks — same granularity as stdlib."""
        results: list[tuple[int, float]] = []
        for i in range(tid, n_items, n_workers):
            cs = workload_fn(BASE_SEED, i)
            results.append((i, cs))
        return results

    items = [(tid, num_items, WORKERS) for tid in range(WORKERS)]

    t0 = time.perf_counter()
    batch_results = parallel_starmap(coarse_worker, items, num_workers=WORKERS)
    elapsed = time.perf_counter() - t0

    checksums = [0.0] * num_items
    for batch in batch_results:
        for idx, cs in batch:
            checksums[idx] = cs
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

    from bench_format import FREE_THREADED
    note = ("Free-threaded mode — true parallelism for pure Python!"
            if FREE_THREADED
            else "GIL active — threads cannot parallelise pure Python code")
    print_header(
        "PURE PYTHON CPU BENCHMARK",
        extra_lines=[f"NOTE         : {note}"],
    )

    for wname in selected:
        wfn = WORKLOAD_FNS[wname]
        wlabel = WORKLOAD_LABELS[wname]
        num_items = WORKLOAD_ITEMS[wname]

        section(f"SEQUENTIAL BASELINE: {wname}")
        print("    Running …", end=" ", flush=True)
        seq_time, _seq_cs = bench_sequential(wfn, num_items)
        print(f"{seq_time:.4f}s")

        print("    [1/2] stdlib threading …", end=" ", flush=True)
        std_elapsed, std_cs = bench_stdlib(wfn, num_items)
        print(f"{std_elapsed:.4f}s")

        print("    [2/2] parallel_starmap …", end=" ", flush=True)
        ct_elapsed, ct_cs = bench_pmap(wfn, num_items)
        print(f"{ct_elapsed:.4f}s")

        # checksum validation
        s_sum = sum(std_cs)
        c_sum = sum(ct_cs)
        tol = max(abs(s_sum) * 1e-9, 1e-9)
        cs_ok = abs(s_sum - c_sum) < tol

        std_m = Metrics(
            label="stdlib", ops=num_items, elapsed=std_elapsed,
            correct=True, seq_elapsed=seq_time,
            extra={"checksums_match": "YES" if cs_ok else "NO"},
        )
        ct_m = Metrics(
            label="cthreading", ops=num_items, elapsed=ct_elapsed,
            correct=cs_ok, seq_elapsed=seq_time,
            extra={"checksums_match": "YES" if cs_ok else "NO"},
        )

        record(SUITE, f"{wname} ({wlabel})", std_m, ct_m, show_seq=True)

    print_suite_summary(SUITE)
    export_json(os.path.join(os.path.dirname(__file__), f"results_{SUITE}.json"))


if __name__ == "__main__":
    main()
