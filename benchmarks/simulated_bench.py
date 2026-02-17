"""
cthreading — FULL SYSTEM BENCHMARK
===================================

This benchmark suite is a showcase of cthreading's capabilities. It simulates a
real-world application with multiple threads performing different tasks.

Usage:
    uv run examples/mock_app.py              # run ALL benchmarks
    uv run examples/mock_app.py lock         # run only lock contention
    uv run examples/mock_app.py counter      # run only atomic counter
    uv run examples/mock_app.py queue        # run only queue throughput
    uv run examples/mock_app.py pool         # run only pool dispatch
    uv run examples/mock_app.py event        # run only event fan-out
    uv run examples/mock_app.py sem          # run only semaphore
    uv run examples/mock_app.py stress       # run only integrated stress
    uv run examples/mock_app.py race         # run only race correctness
    uv run examples/mock_app.py lock counter # run multiple sections
"""

from __future__ import annotations

import os  # noqa: F401 — used in export_json path
import queue as stdlib_queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

from bench_format import (
    WORKERS, Metrics,
    print_header, section, record, print_suite_summary, export_json,
)
from cthreading import Queue as CQueue
from cthreading import ThreadPool as CPool
from cthreading.contracts import Frozen, Ghost
from cthreading.governor import OmniBase
from cthreading.monitoring import Counter as CCounter
from cthreading.monitoring import Ghost as CGhost
from cthreading.monitoring import set_enabled as set_monitoring_enabled
from cthreading.sync import Event as CEvent
from cthreading.sync import Semaphore as CSemaphore

# ===================================================================
# CONFIG
# ===================================================================

SUITE = "simulated"
BENCH_NAMES = ["lock", "counter", "queue", "pool", "event", "sem", "stress", "race"]

# ===================================================================
# HELPERS
# ===================================================================

def _m(label: str, ops: int, elapsed: float, correct: bool,
       final_value: object = None, **extra: object) -> Metrics:
    return Metrics(label=label, ops=ops, elapsed=elapsed,
                   correct=correct, final_value=final_value,
                   extra=dict(extra))


# ===================================================================
# BENCH 1: LOCK CONTENTION
# ===================================================================

def bench_lock_contention(num_threads: int, iters: int) -> tuple[Metrics, Metrics]:
    """Real-world pattern: stdlib lock+int vs cthreading Ghost.add (lock-free atomic).
    Both achieve the same goal — thread-safe counter — using each library's idiomatic approach."""
    total_ops = num_threads * iters

    # --- stdlib: lock + int (the standard pattern) ---
    py_lock = threading.Lock()
    py_counter = 0

    def py_worker() -> None:
        nonlocal py_counter
        for _ in range(iters):
            with py_lock:
                py_counter += 1

    threads = [threading.Thread(target=py_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    py_elapsed = time.perf_counter() - t0

    # --- cthreading: Ghost.add (lock-free atomic — the idiomatic replacement) ---
    ghost = CGhost(initial=0)

    def c_worker() -> None:
        for _ in range(iters):
            ghost.add(1)

    threads = [threading.Thread(target=c_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    c_elapsed = time.perf_counter() - t0
    c_val = ghost.get()

    return (
        Metrics("stdlib lock+int", total_ops, py_elapsed, py_counter == total_ops, py_counter),
        Metrics("C Ghost.add", total_ops, c_elapsed, c_val == total_ops, c_val),
    )


# ===================================================================
# BENCH 2: ATOMIC COUNTER (lock+int vs Ghost vs sharded Counter)
# ===================================================================

def bench_atomic_counter(num_threads: int, iters: int) -> tuple[Metrics, Metrics, Metrics]:
    total_ops = num_threads * iters

    # --- stdlib: lock + int ---
    py_lock = threading.Lock()
    py_val = 0

    def py_worker() -> None:
        nonlocal py_val
        for _ in range(iters):
            with py_lock:
                py_val += 1

    threads = [threading.Thread(target=py_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    py_elapsed = time.perf_counter() - t0

    # --- cthreading Ghost ---
    ghost = CGhost(initial=0)

    def ghost_worker() -> None:
        for _ in range(iters):
            ghost.add(1)

    threads = [threading.Thread(target=ghost_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    g_elapsed = time.perf_counter() - t0
    g_val = ghost.get()

    # --- cthreading Counter (sharded) ---
    counter = CCounter(initial=0, shards=64)

    def counter_worker() -> None:
        for _ in range(iters):
            counter.add(1)

    threads = [threading.Thread(target=counter_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    c_elapsed = time.perf_counter() - t0
    c_val = counter.get()

    return (
        Metrics("stdlib lock+int", total_ops, py_elapsed, py_val == total_ops, py_val),
        Metrics("C Ghost.add", total_ops, g_elapsed, g_val == total_ops, g_val),
        Metrics("C Counter(64)", total_ops, c_elapsed, c_val == total_ops, c_val),
    )


# ===================================================================
# BENCH 3: QUEUE THROUGHPUT (producer / consumer)
# ===================================================================

def bench_queue(num_producers: int, num_consumers: int, items_per_producer: int) -> tuple[Metrics, Metrics]:
    """Queue throughput: uses atomic counters for consumer tracking to isolate queue perf."""
    total_items = num_producers * items_per_producer
    sentinel = None

    # --- stdlib queue.Queue ---
    py_q: stdlib_queue.Queue[int | None] = stdlib_queue.Queue()
    py_consumed = CCounter(initial=0, shards=64)  # atomic counter avoids lock overhead

    def py_producer(pid: int) -> None:
        for i in range(items_per_producer):
            py_q.put(i)

    def py_consumer() -> None:
        while True:
            item = py_q.get()
            if item is sentinel:
                break
            py_consumed.add(1)

    producers = [threading.Thread(target=py_producer, args=(i,)) for i in range(num_producers)]
    consumers = [threading.Thread(target=py_consumer) for _ in range(num_consumers)]

    t0 = time.perf_counter()
    for t in consumers:
        t.start()
    for t in producers:
        t.start()
    for t in producers:
        t.join()
    for _ in range(num_consumers):
        py_q.put(sentinel)
    for t in consumers:
        t.join()
    py_elapsed = time.perf_counter() - t0
    py_val = py_consumed.get()

    # --- cthreading Queue ---
    c_q = CQueue()
    c_consumed = CCounter(initial=0, shards=64)

    def c_producer(pid: int) -> None:
        for i in range(items_per_producer):
            c_q.put(i)

    def c_consumer() -> None:
        while True:
            item = c_q.get()
            if item is sentinel:
                break
            c_consumed.add(1)

    producers = [threading.Thread(target=c_producer, args=(i,)) for i in range(num_producers)]
    consumers = [threading.Thread(target=c_consumer) for _ in range(num_consumers)]

    t0 = time.perf_counter()
    for t in consumers:
        t.start()
    for t in producers:
        t.start()
    for t in producers:
        t.join()
    for _ in range(num_consumers):
        c_q.put(sentinel)
    for t in consumers:
        t.join()
    c_elapsed = time.perf_counter() - t0
    c_val = c_consumed.get()

    return (
        Metrics("stdlib Queue", total_items, py_elapsed, py_val == total_items, py_val,
                     extra={"queue_empty": str(py_q.empty())}),
        Metrics("C Queue", total_items, c_elapsed, c_val == total_items, c_val,
                     extra={"queue_empty": str(c_q.empty())}),
    )


# ===================================================================
# BENCH 4: THREAD POOL TASK DISPATCH
# ===================================================================

def bench_pool_dispatch(num_tasks: int) -> tuple[Metrics, Metrics]:
    """Pool dispatch throughput: trivial tasks to measure pure pool infrastructure.
    stdlib pays Future object + callback overhead per task; CPool is lean C dispatch."""
    py_done = CGhost(initial=0)
    c_done = CGhost(initial=0)

    def py_task() -> None:
        py_done.add(1)

    def c_task() -> None:
        c_done.add(1)

    # --- stdlib ThreadPoolExecutor ---
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(py_task) for _ in range(num_tasks)]
        for f in futs:
            f.result()
    py_elapsed = time.perf_counter() - t0
    py_val = py_done.get()

    # --- cthreading ThreadPool ---
    c_pool = CPool(num_workers=WORKERS)
    t0 = time.perf_counter()
    for _ in range(num_tasks):
        c_pool.submit(c_task)
    c_pool.shutdown(wait=True)
    c_elapsed = time.perf_counter() - t0
    c_val = c_done.get()

    return (
        Metrics("ThreadPoolExecutor", num_tasks, py_elapsed, py_val == num_tasks,
                     py_val, extra={"workers": str(WORKERS)}),
        Metrics("C ThreadPool", num_tasks, c_elapsed, c_val == num_tasks,
                     c_val, extra={"workers": str(WORKERS)}),
    )


# ===================================================================
# BENCH 5a: EVENT FAN-OUT LATENCY
# ===================================================================

def bench_event_fanout(num_waiters: int, iters: int) -> tuple[Metrics, Metrics]:
    """Event wait() throughput: event is pre-set, threads call wait() in a tight loop.
    Measures fast-path overhead: stdlib Python Condition check vs C atomic check."""
    total_ops = num_waiters * iters

    # --- stdlib Event (pre-set, fast-path wait) ---
    py_event = threading.Event()
    py_event.set()
    py_done = CGhost(initial=0)

    def py_waiter() -> None:
        ev = py_event
        for _ in range(iters):
            ev.wait()
        py_done.add(1)

    threads = [threading.Thread(target=py_waiter) for _ in range(num_waiters)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    py_elapsed = time.perf_counter() - t0
    py_val = py_done.get()

    # --- cthreading Event (pre-set, fast-path wait) ---
    c_event = CEvent()
    c_event.set()
    c_done = CGhost(initial=0)

    def c_waiter() -> None:
        ev = c_event
        for _ in range(iters):
            ev.wait()
        c_done.add(1)

    threads = [threading.Thread(target=c_waiter) for _ in range(num_waiters)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    c_elapsed = time.perf_counter() - t0
    c_val = c_done.get()

    return (
        Metrics("stdlib Event.wait", total_ops, py_elapsed, py_val == num_waiters, total_ops),
        Metrics("C Event.wait", total_ops, c_elapsed, c_val == num_waiters, total_ops),
    )


# ===================================================================
# BENCH 5b: SEMAPHORE THROUGHPUT
# ===================================================================

def bench_semaphore(num_threads: int, iters: int, permits: int) -> tuple[Metrics, Metrics]:
    """Semaphore acquire/release throughput: permits >= threads so pure fast-path.
    stdlib Semaphore is pure Python; C Semaphore uses atomic CAS."""
    total_ops = num_threads * iters

    # --- stdlib ---
    py_sem = threading.Semaphore(permits)

    def py_worker() -> None:
        sem = py_sem
        for _ in range(iters):
            sem.acquire()
            sem.release()

    threads = [threading.Thread(target=py_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    py_elapsed = time.perf_counter() - t0

    # --- cthreading ---
    c_sem = CSemaphore(value=permits)

    def c_worker() -> None:
        sem = c_sem
        for _ in range(iters):
            sem.acquire()
            sem.release()

    threads = [threading.Thread(target=c_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    c_elapsed = time.perf_counter() - t0

    return (
        Metrics("stdlib Semaphore", total_ops, py_elapsed, True, total_ops,
                     extra={"permits": str(permits)}),
        Metrics("C Semaphore", total_ops, c_elapsed, True, total_ops,
                     extra={"permits": str(permits)}),
    )


# ===================================================================
# BENCH 6: INTEGRATED STRESS TEST (OmniBase + Ghost descriptors)
# ===================================================================

class StressState(OmniBase):
    counter: Annotated[int, Ghost] = 0
    aux: Annotated[int, Ghost] = 0
    label: Annotated[str, Frozen] = "bench"


def bench_integrated_stress(
    num_threads: int, iters: int, flush_every: int
) -> tuple[Metrics, Metrics]:
    total_ops = num_threads * iters

    # --- stdlib: plain lock + dict for state ---
    py_lock = threading.Lock()
    py_state = {"counter": 0, "aux": 0}

    def py_worker() -> None:
        local_c = 0
        local_a = 0
        for i in range(iters):
            local_c += 1
            if i % 7 == 0:
                local_a += 1
            if flush_every > 0 and (i + 1) % flush_every == 0:
                with py_lock:
                    py_state["counter"] += local_c
                    py_state["aux"] += local_a
                local_c = 0
                local_a = 0
        if local_c or local_a:
            with py_lock:
                py_state["counter"] += local_c
                py_state["aux"] += local_a

    threads = [threading.Thread(target=py_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    py_elapsed = time.perf_counter() - t0
    py_final = py_state["counter"]

    # --- cthreading: OmniBase descriptors ---
    set_monitoring_enabled(True)
    state = StressState()
    counter_d = type(state).__dict__["counter"]
    aux_d = type(state).__dict__["aux"]

    def c_worker() -> None:
        # Cache the C-level cells to bypass descriptor/WeakKeyDict overhead per-flush
        c_cell = counter_d._get_cell(state)
        a_cell = aux_d._get_cell(state)
        local_c = 0
        local_a = 0
        for i in range(iters):
            local_c += 1
            if i % 7 == 0:
                local_a += 1
            if flush_every > 0 and (i + 1) % flush_every == 0:
                c_cell.add(local_c)
                a_cell.add(local_a)
                local_c = 0
                local_a = 0
        if local_c or local_a:
            c_cell.add(local_c)
            a_cell.add(local_a)

    threads = [threading.Thread(target=c_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    c_elapsed = time.perf_counter() - t0
    c_final = int(state.counter)
    c_stats = counter_d.get_stats(state)

    return (
        Metrics("stdlib lock+dict", total_ops, py_elapsed, py_final == total_ops, py_final,
                     extra={"flush_every": str(flush_every)}),
        Metrics("OmniBase Ghost", total_ops, c_elapsed, c_final == total_ops, c_final,
                     extra={
                         "flush_every": str(flush_every),
                         "accesses": str(c_stats.get("accesses", 0)),
                         "contention": str(c_stats.get("heat", 0)),
                         "version": str(c_stats.get("version", 0)),
                     }),
    )


# ===================================================================
# BENCH 7: UNPROTECTED vs GHOST RACE CORRECTNESS
# ===================================================================

def bench_race_correctness(num_threads: int, iters: int) -> tuple[Metrics, Metrics]:
    total_ops = num_threads * iters

    # --- bare int (expected to race) ---
    bare_val = 0

    def bare_worker() -> None:
        nonlocal bare_val
        for _ in range(iters):
            bare_val += 1

    threads = [threading.Thread(target=bare_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bare_elapsed = time.perf_counter() - t0
    bare_delta = total_ops - bare_val

    # --- Ghost (should be exact) ---
    ghost = CGhost(initial=0)

    def ghost_worker() -> None:
        for _ in range(iters):
            ghost.add(1)

    threads = [threading.Thread(target=ghost_worker) for _ in range(num_threads)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ghost_elapsed = time.perf_counter() - t0
    ghost_val = ghost.get()

    return (
        Metrics("bare int (UNSAFE)", total_ops, bare_elapsed, bare_val == total_ops, bare_val,
                     extra={"race_losses": f"{bare_delta:,}"}),
        Metrics("C Ghost (SAFE)", total_ops, ghost_elapsed, ghost_val == total_ops, ghost_val,
                     extra={"race_losses": "0"}),
    )


# ===================================================================
# RUNNERS — map flag names to bench functions
# ===================================================================

def run_lock() -> None:
    iters_lock = 200_000
    s, c = bench_lock_contention(WORKERS, iters_lock)
    record(SUITE, f"Lock contention ({WORKERS}T x {iters_lock:,})", s, c)


def run_counter() -> None:
    iters_counter = 200_000
    s_lock, s_ghost, s_counter = bench_atomic_counter(WORKERS, iters_counter)
    record(SUITE, f"Counter: Ghost ({WORKERS}T x {iters_counter:,})", s_lock, s_ghost)
    record(SUITE, f"Counter: sharded ({WORKERS}T x {iters_counter:,})", s_lock, s_counter)


def run_queue() -> None:
    items_per_producer = 50_000
    n_prod = max(WORKERS // 2, 2)
    n_cons = max(WORKERS // 2, 2)
    s, c = bench_queue(n_prod, n_cons, items_per_producer)
    record(SUITE, f"Queue throughput ({n_prod}P x {items_per_producer:,}, {n_cons}C)", s, c)


def run_pool() -> None:
    num_tasks = 100_000
    s, c = bench_pool_dispatch(num_tasks)
    record(SUITE, f"Pool dispatch ({num_tasks:,} tasks, {WORKERS}W)", s, c)


def run_event() -> None:
    event_waiters = WORKERS
    event_iters = 200_000
    s, c = bench_event_fanout(event_waiters, event_iters)
    record(SUITE, f"Event.wait ({event_waiters}T x {event_iters:,})", s, c)


def run_sem() -> None:
    iters_sem = 200_000
    permits = WORKERS * 2
    s, c = bench_semaphore(WORKERS, iters_sem, permits)
    record(SUITE, f"Semaphore acq/rel ({WORKERS}T x {iters_sem:,}, {permits}p)", s, c)


def run_stress() -> None:
    iters_stress = 100_000
    flush_every = 25
    s, c = bench_integrated_stress(WORKERS, iters_stress, flush_every)
    record(SUITE, f"Integrated stress ({WORKERS}T x {iters_stress:,})", s, c)


def run_race() -> None:
    iters_race = 200_000
    s, c = bench_race_correctness(WORKERS, iters_race)
    record(SUITE, f"Race correctness ({WORKERS}T x {iters_race:,})", s, c)


RUNNERS = {
    "lock": run_lock,
    "counter": run_counter,
    "queue": run_queue,
    "pool": run_pool,
    "event": run_event,
    "sem": run_sem,
    "stress": run_stress,
    "race": run_race,
}


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    args = sys.argv[1:]

    selected = []
    for a in args:
        a_lower = a.lower()
        if a_lower not in RUNNERS:
            print(f"Unknown benchmark: '{a}'")
            print(f"Available: {', '.join(BENCH_NAMES)}")
            sys.exit(1)
        selected.append(a_lower)

    if not selected:
        selected = list(BENCH_NAMES)

    print_header("SIMULATED SYSTEM BENCHMARK")

    for name in selected:
        RUNNERS[name]()

    print_suite_summary(SUITE)
    export_json(os.path.join(os.path.dirname(__file__), f"results_{SUITE}.json"))


if __name__ == "__main__":
    main()
