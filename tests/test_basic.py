# tests/test_basic.py
from __future__ import annotations

import threading
import time
from typing import Annotated

from cthreading import Lock, Queue, ThreadPool, auto_thread
from cthreading.contracts import Ghost
from cthreading.governor import OmniBase
from cthreading.monitoring import Counter
from cthreading.monitoring import Ghost as MonitoringGhost


class ContendedData(OmniBase):
    counter: Annotated[int, Ghost] = 0


def test_ghost_contention() -> None:
    """Original test: Ghost descriptor + threads for correctness."""
    obj = ContendedData()

    def increment() -> None:
        for _ in range(1000):
            obj.counter += 1

    threads = [threading.Thread(target=increment) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ghost_descriptor: Ghost[int] = type(obj).__dict__['counter']
    stats = ghost_descriptor.get_stats(obj)

    print(f"\nFinal Count: {obj.counter}")
    print(f"Stats: {stats}")

    assert obj.counter == 10000
    assert stats["heat"] >= 0


def test_monitoring_ghost() -> None:
    """Test Ghost cell directly."""
    g = MonitoringGhost(initial=0)
    g.add(10)
    assert g.get() == 10
    g.set(42)
    assert g.get() == 42


def test_counter() -> None:
    """Test sharded Counter."""
    c = Counter(initial=0, shards=8)
    for _ in range(1000):
        c.add(1)
    assert c.get() == 1000


def test_lock() -> None:
    """Test C Lock."""
    lock = Lock()
    with lock:
        assert lock.locked()
    assert not lock.locked()


def test_queue() -> None:
    """Test C Queue."""
    q = Queue()
    q.put("hello")
    q.put("world")
    assert q.qsize() == 2
    assert q.get() == "hello"
    assert q.get() == "world"
    assert q.empty()


def test_thread_pool() -> None:
    """Test ThreadPool submit."""
    results: list[int] = []
    lock = threading.Lock()

    def work(i: int) -> None:
        with lock:
            results.append(i)

    pool = ThreadPool(num_workers=2)
    for i in range(10):
        pool.submit(work, (i,))
    pool.shutdown(wait=True)

    for _ in range(50):
        if len(results) >= 10:
            break
        time.sleep(0.1)

    print(f"\nThreadPool results: {len(results)}/10")
    assert len(results) == 10


def test_auto_thread_decorator() -> None:
    """Test auto_thread decorator."""
    counter = MonitoringGhost(initial=0)

    @auto_thread
    def bump(val: int) -> None:
        counter.add(val)

    for _ in range(100):
        bump(1)

    for _ in range(50):
        if counter.get() >= 100:
            break
        time.sleep(0.1)

    print(f"\nauto_thread counter: {counter.get()}")
    assert counter.get() == 100


if __name__ == "__main__":
    test_ghost_contention()
    test_monitoring_ghost()
    test_counter()
    test_lock()
    test_queue()
    test_thread_pool()
    test_auto_thread_decorator()
    print("\nAll tests passed!")
