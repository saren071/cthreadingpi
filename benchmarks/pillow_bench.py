"""
cthreading — PILLOW IMAGE-PROCESSING BENCHMARK
=================================================

stdlib threading.Thread vs cthreading @auto_thread on image-processing
workloads using Pillow.  Pillow's C code releases the GIL for many
filter/resize operations, enabling real parallelism.

The benchmark generates random images in-memory (no disk I/O) and applies
heavy transformations.  Each worker processes its own batch of images.

Workloads:
  A. Gaussian blur     — repeated heavy blur kernel on large images
  B. Resize cascade    — downscale + upscale chain (Lanczos resampling)
  C. Edge detection    — FIND_EDGES + SHARPEN + EMBOSS filter stack
  D. Color transform   — RGB → L → quantize → RGB round-trip
  E. Composite blend   — alpha-composite multiple layers

Usage:
    uv run examples/pillow_bench.py              # run ALL
    uv run examples/pillow_bench.py blur         # single workload
    uv run examples/pillow_bench.py resize edge  # multiple
"""

from __future__ import annotations

import hashlib
import os
import random
import struct
import sys
import threading
import time
from typing import TYPE_CHECKING

from bench_format import (
    WORKERS,
    Metrics,
    export_json,
    print_header,
    print_suite_summary,
    record,
    section,
)
from PIL import Image, ImageFilter

from cthreading import parallel_starmap

if TYPE_CHECKING:
    from collections.abc import Callable

# ===================================================================
# CONFIG
# ===================================================================

SUITE = "pillow"
NUM_TASKS = WORKERS
BASE_SEED = 99999

# Image sizes / repetitions — tuned for ~1-4 s per worker on modern CPU.
IMG_SIZE = 2048           # pixels per side for generated images
BLUR_REPS = 12            # how many blur passes per image
RESIZE_REPS = 8           # how many resize round-trips
EDGE_REPS = 10            # filter stack applications
COLOR_REPS = 6            # colour-space round-trips
COMPOSITE_LAYERS = 8      # number of layers to alpha-composite
IMAGES_PER_WORKER = 2     # images each worker processes

WORKLOAD_ORDER = ["blur", "resize", "edge", "color", "composite"]

# ===================================================================
# IMAGE GENERATION (deterministic, in-memory)
# ===================================================================

def make_random_image(seed: int, size: int = IMG_SIZE) -> Image.Image:
    """Generate a deterministic random RGB image entirely in memory."""
    rng = random.Random(seed)
    data = rng.randbytes(size * size * 3)
    img = Image.frombytes("RGB", (size, size), data)
    return img


def image_checksum(img: Image.Image) -> float:
    """Fast hash-based checksum of image data."""
    h = hashlib.md5(img.tobytes()).digest()
    return float(struct.unpack("<d", h[:8])[0])


# ===================================================================
# WORKLOAD FUNCTIONS
# ===================================================================
# Each takes (seed, chunk_id) and returns a checksum float.

def workload_blur(seed: int, chunk_id: int) -> float:
    """Repeated Gaussian blur — heavy convolution kernel."""
    cs = 0.0
    for img_i in range(IMAGES_PER_WORKER):
        img = make_random_image(seed + chunk_id * 1000 + img_i)
        for _ in range(BLUR_REPS):
            img = img.filter(ImageFilter.GaussianBlur(radius=5))
        cs += image_checksum(img)
    return cs


def workload_resize(seed: int, chunk_id: int) -> float:
    """Resize cascade — Lanczos downscale then upscale, repeated."""
    cs = 0.0
    for img_i in range(IMAGES_PER_WORKER):
        img = make_random_image(seed + chunk_id * 1000 + img_i)
        w, h = img.size
        for _ in range(RESIZE_REPS):
            img = img.resize((w // 4, h // 4), Image.Resampling.LANCZOS)
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        cs += image_checksum(img)
    return cs


def workload_edge(seed: int, chunk_id: int) -> float:
    """Edge detection + sharpen + emboss filter stack."""
    cs = 0.0
    for img_i in range(IMAGES_PER_WORKER):
        img = make_random_image(seed + chunk_id * 1000 + img_i)
        for _ in range(EDGE_REPS):
            img = img.filter(ImageFilter.FIND_EDGES)
            img = img.filter(ImageFilter.SHARPEN)
            img = img.filter(ImageFilter.EMBOSS)
        cs += image_checksum(img)
    return cs


def workload_color(seed: int, chunk_id: int) -> float:
    """Colour-space round-trip: RGB → L → quantize → RGB."""
    cs = 0.0
    for img_i in range(IMAGES_PER_WORKER):
        img = make_random_image(seed + chunk_id * 1000 + img_i)
        for _ in range(COLOR_REPS):
            grey = img.convert("L")
            quantized = grey.quantize(colors=64)
            img = quantized.convert("RGB")
        cs += image_checksum(img)
    return cs


def workload_composite(seed: int, chunk_id: int) -> float:
    """Alpha-composite multiple random layers."""
    cs = 0.0
    for img_i in range(IMAGES_PER_WORKER):
        base = make_random_image(seed + chunk_id * 1000 + img_i).convert("RGBA")
        for layer_i in range(COMPOSITE_LAYERS):
            layer_seed = seed + chunk_id * 1000 + img_i * 100 + layer_i
            overlay = make_random_image(layer_seed, IMG_SIZE).convert("RGBA")
            # Set varying alpha
            alpha = overlay.split()[3].point(lambda p: int(p) // 4)
            overlay.putalpha(alpha)
            base = Image.alpha_composite(base, overlay)
        cs += image_checksum(base.convert("RGB"))
    return cs


WORKLOAD_FNS: dict[str, Callable[[int, int], float]] = {
    "blur":      workload_blur,
    "resize":    workload_resize,
    "edge":      workload_edge,
    "color":     workload_color,
    "composite": workload_composite,
}

WORKLOAD_LABELS: dict[str, str] = {
    "blur":      f"Gaussian blur {IMG_SIZE}x{IMG_SIZE}, {BLUR_REPS} passes x {IMAGES_PER_WORKER} imgs",
    "resize":    f"Lanczos resize cascade {IMG_SIZE}x{IMG_SIZE}, {RESIZE_REPS} trips x {IMAGES_PER_WORKER} imgs",
    "edge":      f"Edge+Sharpen+Emboss {IMG_SIZE}x{IMG_SIZE}, {EDGE_REPS} passes x {IMAGES_PER_WORKER} imgs",
    "color":     f"Color-space round-trip {IMG_SIZE}x{IMG_SIZE}, {COLOR_REPS} trips x {IMAGES_PER_WORKER} imgs",
    "composite": f"Alpha composite {COMPOSITE_LAYERS} layers {IMG_SIZE}x{IMG_SIZE} x {IMAGES_PER_WORKER} imgs",
}


# ===================================================================
# SEQUENTIAL BASELINE
# ===================================================================

def bench_sequential(
    workload_fn: Callable[[int, int], float],
    num_workers: int,
) -> tuple[float, list[float]]:
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

    from PIL import __version__ as pil_ver
    print_header(
        "PILLOW IMAGE-PROCESSING BENCHMARK",
        extra_lines=[
            f"Pillow       : {pil_ver}",
            f"Image size   : {IMG_SIZE}x{IMG_SIZE} RGB",
            f"Imgs/task    : {IMAGES_PER_WORKER}",
        ],
    )

    for wname in selected:
        wfn = WORKLOAD_FNS[wname]
        wlabel = WORKLOAD_LABELS[wname]

        section(f"SEQUENTIAL BASELINE: {wname}")
        print("    Running …", end=" ", flush=True)
        seq_time, _ = bench_sequential(wfn, NUM_TASKS)
        print(f"{seq_time:.4f}s")

        print("    [1/2] stdlib threading …", end=" ", flush=True)
        std_elapsed, std_cs = bench_stdlib(wfn, NUM_TASKS)
        print(f"{std_elapsed:.4f}s")

        print("    [2/2] parallel_starmap …", end=" ", flush=True)
        ct_elapsed, ct_cs = bench_pmap(wfn, NUM_TASKS)
        print(f"{ct_elapsed:.4f}s")

        # checksum validation
        s_sorted = sorted(std_cs)
        c_sorted = sorted(ct_cs)
        cs_ok = len(s_sorted) == len(c_sorted) and all(
            abs(a - b) < max(abs(a) * 1e-9, 1e-9)
            for a, b in zip(s_sorted, c_sorted, strict=False)
        )

        std_m = Metrics(
            label="stdlib", ops=NUM_TASKS, elapsed=std_elapsed,
            correct=True, seq_elapsed=seq_time,
            extra={"checksums_match": "YES" if cs_ok else "NO"},
        )
        ct_m = Metrics(
            label="cthreading", ops=NUM_TASKS, elapsed=ct_elapsed,
            correct=cs_ok, seq_elapsed=seq_time,
            extra={"checksums_match": "YES" if cs_ok else "NO"},
        )

        record(SUITE, f"{wname} ({wlabel})", std_m, ct_m, show_seq=True)

    print_suite_summary(SUITE)
    export_json(os.path.join(os.path.dirname(__file__), f"results_{SUITE}.json"))


if __name__ == "__main__":
    main()
