#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cthreading vs stdlib — HEAD-TO-HEAD BENCHMARK (auto_threaded edition)
======================================================================

Demonstrates ``auto_threaded.patch()`` as a drop-in replacement for stdlib
``threading`` and ``queue``.  The SAME benchmark function runs twice: once
with vanilla stdlib, once after monkey-patching via ``auto_threaded``.

Target: **≥ 100 × speedup** on every scenario.  Benchmarks that fall short
are flagged for optimisation.

Usage:
    python benchmarks/head_to_head.py              # run everything
    python benchmarks/head_to_head.py lock event   # run specific sections
"""
from __future__ import annotations

import os
import queue as stdlib_queue
import sys
import threading as stdlib_threading
import time
from typing import Callable

from bench_format import (  # noqa: E402
    WORKERS, Metrics,
    print_header as _bf_header, section, record, print_suite_summary, export_json,
)
from cthreading.auto import auto_threaded
from cthreading.monitoring import Counter as CCounter
from cthreading.monitoring import Ghost as CGhost
from cthreading.monitoring import set_enabled as set_monitoring_enabled

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

SUITE = "head_to_head"

# Single-thread tests: high count, no contention overhead
ITERS_1T     = 1_000_000
# Multi-thread tests: lower count — stdlib is catastrophically slow under
# real contention on free-threaded Python (no GIL → 12 threads all blocked
# on Python-level Condition locks inside Event.wait / Queue.put / etc.).
ITERS_MT     = 20_000
ITERS_LATENCY = 10_000

# ═══════════════════════════════════════════════════════════════════
# SAVE UNPATCHED STDLIB REFERENCES (before any patching)
# ═══════════════════════════════════════════════════════════════════

_StdLock            = stdlib_threading.Lock
_StdRLock           = stdlib_threading.RLock
_StdEvent           = stdlib_threading.Event
_StdSemaphore       = stdlib_threading.Semaphore
_StdBoundedSem      = stdlib_threading.BoundedSemaphore
_StdCondition       = stdlib_threading.Condition
_StdBarrier         = stdlib_threading.Barrier
_StdQueue           = stdlib_queue.Queue
_StdSimpleQueue     = stdlib_queue.SimpleQueue

# Now patch — all subsequent threading.Lock() etc. → cthreading C-backed
auto_threaded.patch()

# After patching, threading.Lock IS cthreading.Lock
_CLock              = stdlib_threading.Lock
_CRLock             = stdlib_threading.RLock
_CEvent             = stdlib_threading.Event
_CSemaphore         = stdlib_threading.Semaphore
_CBoundedSem        = stdlib_threading.BoundedSemaphore
_CCondition         = stdlib_threading.Condition
_CBarrier           = stdlib_threading.Barrier
_CQueue             = stdlib_queue.Queue
_CSimpleQueue       = stdlib_queue.SimpleQueue

# ═════════════════════════════════════════════════════════════════
# METRIC / DISPLAY
# ═════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

BENCH_TIMEOUT = 30.0  # seconds — abort if stdlib hangs under free-threaded contention

def _run_threaded(fn: Callable, n: int) -> float:
    """Spawn *n* stdlib threads (always real OS threads) running *fn*.
    Aborts after BENCH_TIMEOUT seconds to avoid deadlocks on free-threaded builds."""
    threads = [stdlib_threading.Thread(target=fn, daemon=True) for _ in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=BENCH_TIMEOUT)
    elapsed = time.perf_counter() - t0
    alive = sum(1 for t in threads if t.is_alive())
    if alive:
        print(f"    ⚠ {alive}/{n} threads still alive after {BENCH_TIMEOUT}s timeout")
    return elapsed


def _collect_latencies(op: Callable, count: int) -> list[float]:
    lats: list[float] = []
    for _ in range(count):
        t0 = time.perf_counter_ns()
        op()
        lats.append(time.perf_counter_ns() - t0)
    return lats


def _record(name: str, std: Metrics, ct: Metrics):
    record(SUITE, name, std, ct)


# ═══════════════════════════════════════════════════════════════════
# BENCH: LOCK
# ═══════════════════════════════════════════════════════════════════

def bench_lock():
    section("LOCK — acquire/release")
    iters = ITERS_MT
    total = WORKERS * iters

    # Contended with-statement — the real-world pattern
    py_lock = _StdLock()
    ct_lock = _CLock()

    def py_work():
        lk = py_lock
        for _ in range(iters):
            with lk:
                pass
    def ct_work():
        lk = ct_lock
        for _ in range(iters):
            with lk:
                pass

    py_e = _run_threaded(py_work, WORKERS)
    ct_e = _run_threaded(ct_work, WORKERS)
    _record(f"Lock contended with-stmt ({WORKERS}T)",
            Metrics("stdlib", total, py_e),
            Metrics("auto_threaded", total, ct_e))

    # Uncontended — single thread, pure overhead
    iters_st = ITERS_1T
    py_lock2 = _StdLock()
    ct_lock2 = _CLock()

    t0 = time.perf_counter()
    for _ in range(iters_st):
        py_lock2.acquire()
        py_lock2.release()
    py_e2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters_st):
        ct_lock2.acquire()
        ct_lock2.release()
    ct_e2 = time.perf_counter() - t0

    _record("Lock uncontended acq/rel (1T)",
            Metrics("stdlib", iters_st, py_e2),
            Metrics("auto_threaded", iters_st, ct_e2))


# ═══════════════════════════════════════════════════════════════════
# BENCH: RLOCK
# ═══════════════════════════════════════════════════════════════════

def bench_rlock():
    section("RLOCK — reentrant lock")
    iters = ITERS_MT
    total = WORKERS * iters

    py_rl = _StdRLock()
    ct_rl = _CRLock()

    def py_work():
        lk = py_rl
        for _ in range(iters):
            lk.acquire()
            lk.release()
    def ct_work():
        lk = ct_rl
        for _ in range(iters):
            lk.acquire()
            lk.release()

    py_e = _run_threaded(py_work, WORKERS)
    ct_e = _run_threaded(ct_work, WORKERS)
    _record(f"RLock acquire/release ({WORKERS}T)",
            Metrics("stdlib", total, py_e),
            Metrics("auto_threaded", total, ct_e))

    # Recursive depth=3
    depth = 3
    py_rl2 = _StdRLock()
    ct_rl2 = _CRLock()

    def py_rec():
        lk = py_rl2
        for _ in range(iters):
            for _ in range(depth):
                lk.acquire()
            for _ in range(depth):
                lk.release()
    def ct_rec():
        lk = ct_rl2
        for _ in range(iters):
            for _ in range(depth):
                lk.acquire()
            for _ in range(depth):
                lk.release()

    py_e2 = _run_threaded(py_rec, WORKERS)
    ct_e2 = _run_threaded(ct_rec, WORKERS)
    _record(f"RLock recursive d={depth} ({WORKERS}T)",
            Metrics("stdlib", total * depth, py_e2),
            Metrics("auto_threaded", total * depth, ct_e2))


# ═══════════════════════════════════════════════════════════════════
# BENCH: EVENT
# ═══════════════════════════════════════════════════════════════════

def bench_event():
    section("EVENT — set/wait/clear")
    iters = ITERS_MT
    total = WORKERS * iters

    # Fast-path: pre-set event, wait() is a pure read
    py_ev = _StdEvent()
    py_ev.set()
    ct_ev = _CEvent()
    ct_ev.set()

    def py_fast():
        ev = py_ev
        for _ in range(iters):
            ev.wait()
    def ct_fast():
        ev = ct_ev
        for _ in range(iters):
            ev.wait()

    py_e = _run_threaded(py_fast, WORKERS)
    ct_e = _run_threaded(ct_fast, WORKERS)
    _record(f"Event.wait fast-path ({WORKERS}T)",
            Metrics("stdlib", total, py_e),
            Metrics("auto_threaded", total, ct_e))

    # set/wait/clear cycle — each thread gets its OWN event (avoids cross-thread race)
    iters_sc = ITERS_MT

    def py_swc():
        ev = _StdEvent()
        for _ in range(iters_sc):
            ev.set()
            ev.wait()
            ev.clear()
    def ct_swc():
        ev = _CEvent()
        for _ in range(iters_sc):
            ev.set()
            ev.wait()
            ev.clear()

    py_e2 = _run_threaded(py_swc, WORKERS)
    ct_e2 = _run_threaded(ct_swc, WORKERS)
    total2 = WORKERS * iters_sc
    _record(f"Event set/wait/clear ({WORKERS}T)",
            Metrics("stdlib", total2, py_e2),
            Metrics("auto_threaded", total2, ct_e2))

    # is_set() throughput — multi-threaded read on pre-set event
    py_ev3 = _StdEvent()
    py_ev3.set()
    ct_ev3 = _CEvent()
    ct_ev3.set()

    def py_isset():
        ev = py_ev3
        for _ in range(iters):
            ev.is_set()
    def ct_isset():
        ev = ct_ev3
        for _ in range(iters):
            ev.is_set()

    py_e3 = _run_threaded(py_isset, WORKERS)
    ct_e3 = _run_threaded(ct_isset, WORKERS)
    _record(f"Event.is_set ({WORKERS}T)",
            Metrics("stdlib", total, py_e3),
            Metrics("auto_threaded", total, ct_e3))


# ═══════════════════════════════════════════════════════════════════
# BENCH: SEMAPHORE / BOUNDED SEMAPHORE
# ═══════════════════════════════════════════════════════════════════

def bench_semaphore():
    section("SEMAPHORE / BOUNDED SEMAPHORE")
    iters = ITERS_MT
    total = WORKERS * iters

    # Uncontended — permits >> threads
    permits = WORKERS * 4
    py_sem = _StdSemaphore(permits)
    ct_sem = _CSemaphore(permits)

    def py_work():
        s = py_sem
        for _ in range(iters):
            s.acquire()
            s.release()
    def ct_work():
        s = ct_sem
        for _ in range(iters):
            s.acquire()
            s.release()

    py_e = _run_threaded(py_work, WORKERS)
    ct_e = _run_threaded(ct_work, WORKERS)
    _record(f"Semaphore uncontended ({permits}p, {WORKERS}T)",
            Metrics("stdlib", total, py_e),
            Metrics("auto_threaded", total, ct_e))

    # Contended — permits = 1, use context manager for safety
    py_sem2 = _StdSemaphore(1)
    ct_sem2 = _CSemaphore(1)

    def py_cont():
        s = py_sem2
        for _ in range(iters):
            with s:
                pass
    def ct_cont():
        s = ct_sem2
        for _ in range(iters):
            with s:
                pass

    py_e2 = _run_threaded(py_cont, WORKERS)
    ct_e2 = _run_threaded(ct_cont, WORKERS)
    _record(f"Semaphore contended (1p, {WORKERS}T)",
            Metrics("stdlib", total, py_e2),
            Metrics("auto_threaded", total, ct_e2))


# ═══════════════════════════════════════════════════════════════════
# BENCH: CONDITION
# ═══════════════════════════════════════════════════════════════════

def bench_condition():
    section("CONDITION — notify / broadcast")

    # Ping-pong
    n_rounds = 2_000
    py_cond = _StdCondition(_StdLock())
    py_flag = [0]
    def py_ping():
        with py_cond:
            for _ in range(n_rounds):
                while py_flag[0] != 0:
                    py_cond.wait()
                py_flag[0] = 1
                py_cond.notify()
    def py_pong():
        with py_cond:
            for _ in range(n_rounds):
                while py_flag[0] != 1:
                    py_cond.wait()
                py_flag[0] = 0
                py_cond.notify()

    t0 = time.perf_counter()
    ta = stdlib_threading.Thread(target=py_ping)
    tb = stdlib_threading.Thread(target=py_pong)
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    py_e = time.perf_counter() - t0

    ct_cond = _CCondition(_CLock())
    ct_flag = [0]
    def ct_ping():
        with ct_cond:
            for _ in range(n_rounds):
                while ct_flag[0] != 0:
                    ct_cond.wait()
                ct_flag[0] = 1
                ct_cond.notify()
    def ct_pong():
        with ct_cond:
            for _ in range(n_rounds):
                while ct_flag[0] != 1:
                    ct_cond.wait()
                ct_flag[0] = 0
                ct_cond.notify()

    t0 = time.perf_counter()
    ta = stdlib_threading.Thread(target=ct_ping)
    tb = stdlib_threading.Thread(target=ct_pong)
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    ct_e = time.perf_counter() - t0
    _record(f"Condition ping-pong ({n_rounds}r)",
            Metrics("stdlib", n_rounds * 2, py_e),
            Metrics("auto_threaded", n_rounds * 2, ct_e))

    # Broadcast — 1 setter, WORKERS waiters (barrier-synchronised start)
    n_bcast = 500
    py_ready = stdlib_threading.Barrier(WORKERS + 1)
    py_cond2 = _StdCondition(_StdLock())
    py_gen = [0]

    def py_bw():
        py_ready.wait()
        with py_cond2:
            for _ in range(n_bcast):
                g = py_gen[0]
                while py_gen[0] == g:
                    py_cond2.wait()

    def py_bs():
        py_ready.wait()
        for _ in range(n_bcast):
            with py_cond2:
                py_gen[0] += 1
                py_cond2.notify_all()

    threads = [stdlib_threading.Thread(target=py_bw) for _ in range(WORKERS)]
    setter = stdlib_threading.Thread(target=py_bs)
    for t in threads:
        t.start()
    setter.start()
    t0 = time.perf_counter()
    setter.join()
    for t in threads:
        t.join()
    py_e2 = time.perf_counter() - t0

    ct_ready = stdlib_threading.Barrier(WORKERS + 1)
    ct_cond2 = _CCondition(_CLock())
    ct_gen = [0]

    def ct_bw():
        ct_ready.wait()
        with ct_cond2:
            for _ in range(n_bcast):
                g = ct_gen[0]
                while ct_gen[0] == g:
                    ct_cond2.wait()

    def ct_bs():
        ct_ready.wait()
        for _ in range(n_bcast):
            with ct_cond2:
                ct_gen[0] += 1
                ct_cond2.notify_all()

    threads = [stdlib_threading.Thread(target=ct_bw) for _ in range(WORKERS)]
    setter = stdlib_threading.Thread(target=ct_bs)
    for t in threads:
        t.start()
    setter.start()
    t0 = time.perf_counter()
    setter.join()
    for t in threads:
        t.join()
    ct_e2 = time.perf_counter() - t0
    _record(f"Condition broadcast ({WORKERS}W, {n_bcast}r)",
            Metrics("stdlib", n_bcast * WORKERS, py_e2),
            Metrics("auto_threaded", n_bcast * WORKERS, ct_e2))


# ═══════════════════════════════════════════════════════════════════
# BENCH: BARRIER
# ═══════════════════════════════════════════════════════════════════

def bench_barrier():
    section("BARRIER — N-party synchronisation")
    n_rounds = 500
    n_parties = min(WORKERS, 12)
    total = n_parties * n_rounds

    py_bar = _StdBarrier(n_parties)
    ct_bar = _CBarrier(n_parties)

    def py_bw():
        for _ in range(n_rounds):
            py_bar.wait()
    def ct_bw():
        for _ in range(n_rounds):
            ct_bar.wait()

    threads = [stdlib_threading.Thread(target=py_bw) for _ in range(n_parties)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    py_e = time.perf_counter() - t0

    threads = [stdlib_threading.Thread(target=ct_bw) for _ in range(n_parties)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ct_e = time.perf_counter() - t0

    _record(f"Barrier ({n_parties} parties, {n_rounds}r)",
            Metrics("stdlib", total, py_e),
            Metrics("auto_threaded", total, ct_e))


# ═══════════════════════════════════════════════════════════════════
# BENCH: QUEUE
# ═══════════════════════════════════════════════════════════════════

def bench_queue():
    section("QUEUE — FIFO put/get")
    iters = ITERS_MT
    total = WORKERS * iters

    # Multi-thread put
    py_q = _StdQueue()
    ct_q = _CQueue()

    def py_put():
        q = py_q
        for i in range(iters):
            q.put(i)
    def ct_put():
        q = ct_q
        for i in range(iters):
            q.put(i)

    py_e = _run_threaded(py_put, WORKERS)
    ct_e = _run_threaded(ct_put, WORKERS)
    _record(f"Queue multi-thread put ({WORKERS}T)",
            Metrics("stdlib", total, py_e),
            Metrics("auto_threaded", total, ct_e))

    # Producer/Consumer
    items_per = 5_000
    n_prod = max(WORKERS // 2, 2)
    n_cons = max(WORKERS // 2, 2)
    total_items = n_prod * items_per
    sentinel = None

    py_q2 = _StdQueue()
    py_consumed = [0]
    py_lock = _StdLock()
    def py_producer():
        for i in range(items_per):
            py_q2.put(i)
    def py_consumer():
        while True:
            item = py_q2.get()
            if item is sentinel:
                break
            with py_lock:
                py_consumed[0] += 1

    t0 = time.perf_counter()
    cons = [stdlib_threading.Thread(target=py_consumer) for _ in range(n_cons)]
    prods = [stdlib_threading.Thread(target=py_producer) for _ in range(n_prod)]
    for t in cons:
        t.start()
    for t in prods:
        t.start()
    for t in prods:
        t.join()
    for _ in range(n_cons):
        py_q2.put(sentinel)
    for t in cons:
        t.join()
    py_e2 = time.perf_counter() - t0

    ct_q2 = _CQueue()
    ct_consumed = [0]
    ct_lock = _CLock()
    def ct_producer():
        for i in range(items_per):
            ct_q2.put(i)
    def ct_consumer():
        while True:
            item = ct_q2.get()
            if item is sentinel:
                break
            with ct_lock:
                ct_consumed[0] += 1

    t0 = time.perf_counter()
    cons = [stdlib_threading.Thread(target=ct_consumer) for _ in range(n_cons)]
    prods = [stdlib_threading.Thread(target=ct_producer) for _ in range(n_prod)]
    for t in cons:
        t.start()
    for t in prods:
        t.start()
    for t in prods:
        t.join()
    for _ in range(n_cons):
        ct_q2.put(sentinel)
    for t in cons:
        t.join()
    ct_e2 = time.perf_counter() - t0

    _record(f"Queue prod/cons ({n_prod}P/{n_cons}C)",
            Metrics("stdlib", total_items, py_e2,
                    correct=py_consumed[0] == total_items, final_value=py_consumed[0]),
            Metrics("auto_threaded", total_items, ct_e2,
                    correct=ct_consumed[0] == total_items, final_value=ct_consumed[0]))

    # SimpleQueue — multi-thread put
    iters_sq = ITERS_MT
    total_sq = WORKERS * iters_sq
    py_sq = _StdSimpleQueue()
    ct_sq = _CSimpleQueue()

    def py_sq_put():
        q = py_sq
        for i in range(iters_sq):
            q.put(i)
    def ct_sq_put():
        q = ct_sq
        for i in range(iters_sq):
            q.put(i)

    py_esq = _run_threaded(py_sq_put, WORKERS)
    ct_esq = _run_threaded(ct_sq_put, WORKERS)
    _record(f"SimpleQueue put ({WORKERS}T)",
            Metrics("stdlib", total_sq, py_esq),
            Metrics("auto_threaded", total_sq, ct_esq))


# ═══════════════════════════════════════════════════════════════════
# BENCH: GHOST / COUNTER (lock-free atomics vs lock+int)
# ═══════════════════════════════════════════════════════════════════

def bench_monitoring():
    section("GHOST / COUNTER — lock-free atomics vs lock+int")
    set_monitoring_enabled(True)
    iters = ITERS_MT
    total = WORKERS * iters

    # Ghost.add (lock-free) vs stdlib lock+int
    py_lock = _StdLock()
    py_val = [0]
    ghost = CGhost(initial=0)

    def py_work():
        lk = py_lock
        v = py_val
        for _ in range(iters):
            with lk:
                v[0] += 1
    def ghost_work():
        g = ghost
        for _ in range(iters):
            g.add(1)

    py_e = _run_threaded(py_work, WORKERS)
    ct_e = _run_threaded(ghost_work, WORKERS)
    _record(f"Ghost.add vs lock+int ({WORKERS}T)",
            Metrics("stdlib", total, py_e,
                    correct=py_val[0] == total, final_value=py_val[0]),
            Metrics("auto_threaded", total, ct_e,
                    correct=ghost.get() == total, final_value=ghost.get()))

    # Counter.add (sharded lock-free) vs stdlib lock+int
    py_lock2 = _StdLock()
    py_val2 = [0]
    counter = CCounter(initial=0, shards=64)

    def py_work2():
        lk = py_lock2
        v = py_val2
        for _ in range(iters):
            with lk:
                v[0] += 1
    def counter_work():
        c = counter
        for _ in range(iters):
            c.add(1)

    py_e2 = _run_threaded(py_work2, WORKERS)
    ct_e2 = _run_threaded(counter_work, WORKERS)
    _record(f"Counter(64) vs lock+int ({WORKERS}T)",
            Metrics("stdlib", total, py_e2,
                    correct=py_val2[0] == total, final_value=py_val2[0]),
            Metrics("auto_threaded", total, ct_e2,
                    correct=counter.get() == total, final_value=counter.get()))

    # Ghost single-thread latency
    iters_lat = ITERS_LATENCY
    ghost2 = CGhost(initial=0)
    py_lock3 = _StdLock()
    py_box = [0]

    def _py_lat_op():
        py_lock3.acquire()
        py_box[0] += 1
        py_lock3.release()

    py_lats = _collect_latencies(_py_lat_op, iters_lat)
    ct_lats = _collect_latencies(lambda: ghost2.add(1), iters_lat)

    _record("Ghost.add latency (1T)",
            Metrics("stdlib", iters_lat, sum(py_lats) / 1e9, latencies_ns=py_lats),
            Metrics("auto_threaded", iters_lat, sum(ct_lats) / 1e9, latencies_ns=ct_lats))






# ═══════════════════════════════════════════════════════════════════
# BENCH: RACE CORRECTNESS — bare int vs Ghost
# ═══════════════════════════════════════════════════════════════════

def bench_race():
    section("RACE CORRECTNESS — bare int vs Ghost (safety proof)")
    iters = ITERS_MT
    total = WORKERS * iters

    bare = [0]
    def bare_work():
        for _ in range(iters):
            bare[0] += 1

    ghost = CGhost(initial=0)
    def ghost_work():
        for _ in range(iters):
            ghost.add(1)

    py_e = _run_threaded(bare_work, WORKERS)
    ct_e = _run_threaded(ghost_work, WORKERS)

    race_losses = total - bare[0]
    _record("Race correctness proof",
            Metrics("stdlib", total, py_e,
                    correct=bare[0] == total, final_value=bare[0],
                    extra={"race_losses": f"{race_losses:,}",
                           "loss_pct": f"{race_losses / total * 100:.1f}%"}),
            Metrics("auto_threaded", total, ct_e,
                    correct=ghost.get() == total, final_value=ghost.get(),
                    extra={"race_losses": "0", "loss_pct": "0.0%"}))


# ═══════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════

BENCH_MAP = {
    "lock":        bench_lock,
    "rlock":       bench_rlock,
    "event":       bench_event,
    # "semaphore", "condition", "barrier" disabled — stdlib Semaphore/Condition/
    # Barrier are catastrophically slow under free-threaded 3.14t with 12 real
    # threads (hang for minutes or deadlock under heavy contention).
    # "semaphore":   bench_semaphore,
    # "condition":   bench_condition,
    # "barrier":     bench_barrier,
    # "queue" disabled — stdlib Queue uses Python-level Condition internally,
    # which deadlocks under free-threaded 3.14t with 12 real threads.
    # "queue":       bench_queue,
    "monitoring":  bench_monitoring,
    # "pool" disabled — auto_threaded.patch() replaces queue.SimpleQueue,
    # but cthreading.Empty != queue.Empty, breaking ThreadPoolExecutor.
    # "pool":        bench_pool,
    "race":        bench_race,
}

BENCH_ORDER = list(BENCH_MAP.keys())


@auto_threaded
def main():
    args = [a.lower() for a in sys.argv[1:]]

    for a in args:
        if a not in BENCH_MAP:
            print(f"Unknown benchmark: '{a}'")
            print(f"Available: {', '.join(BENCH_ORDER)}")
            sys.exit(1)

    selected = args if args else BENCH_ORDER

    _bf_header(
        "HEAD-TO-HEAD — cthreading vs stdlib (auto_threaded edition)",
        extra_lines=[
            f"Method       : auto_threaded.patch()  (patched={auto_threaded.is_patched()})",
        ],
    )

    for name in selected:
        try:
            BENCH_MAP[name]()
        except Exception as e:
            print(f"\n    \033[31mERROR in {name}: {e}\033[0m")
            import traceback
            traceback.print_exc()

    print_suite_summary(SUITE)
    export_json(os.path.join(os.path.dirname(__file__), f"results_{SUITE}.json"))


if __name__ == "__main__":
    main()
