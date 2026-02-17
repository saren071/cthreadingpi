"""Tests for all newly implemented cthreading features."""
import time
import threading as stdlib_threading

import cthreading


# ── BoundedSemaphore ──────────────────────────────────────────────

def test_bounded_semaphore_basic():
    bs = cthreading.BoundedSemaphore(2)
    assert bs.acquire()
    assert bs.acquire()
    bs.release()
    bs.release()


def test_bounded_semaphore_over_release():
    bs = cthreading.BoundedSemaphore(1)
    bs.acquire()
    bs.release()
    try:
        bs.release()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── Barrier ───────────────────────────────────────────────────────

def test_barrier_basic():
    results: list[tuple[int, int]] = []
    b = cthreading.Barrier(3)

    def worker(i: int) -> None:
        idx = b.wait()
        results.append((i, idx))

    threads = [stdlib_threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 3


def test_barrier_action():
    action_called: list[bool] = []
    b = cthreading.Barrier(2, action=lambda: action_called.append(True))

    def worker():
        b.wait()

    t = stdlib_threading.Thread(target=worker)
    t.start()
    b.wait()
    t.join()
    assert action_called == [True]


def test_barrier_abort():
    b = cthreading.Barrier(2)
    b.abort()
    try:
        b.wait()
        assert False, "Should have raised BrokenBarrierError"
    except cthreading.BrokenBarrierError:
        pass


def test_barrier_properties():
    b = cthreading.Barrier(5)
    assert b.parties == 5
    assert b.n_waiting == 0
    assert b.broken is False


# ── Condition.wait_for ────────────────────────────────────────────

def test_condition_wait_for():
    lock = cthreading.Lock()
    cond = cthreading.Condition(lock)
    flag = [False]

    def setter():
        time.sleep(0.05)
        with cond:
            flag[0] = True
            cond.notify_all()

    t = stdlib_threading.Thread(target=setter)
    t.start()
    with cond:
        result = cond.wait_for(lambda: flag[0], timeout=2.0)
    assert result is True
    t.join()


def test_condition_wait_for_timeout():
    lock = cthreading.Lock()
    cond = cthreading.Condition(lock)
    with cond:
        result = cond.wait_for(lambda: False, timeout=0.05)
    assert result is False


# ── Queue exceptions + new methods ───────────────────────────────

def test_queue_full_exception():
    q = cthreading.Queue(maxsize=1)
    q.put("a")
    try:
        q.put_nowait("b")
        assert False, "Should have raised Full"
    except cthreading.Full:
        pass
    q.get()


def test_queue_empty_exception():
    q = cthreading.Queue()
    try:
        q.get_nowait()
        assert False, "Should have raised Empty"
    except cthreading.Empty:
        pass


def test_queue_timeout():
    q = cthreading.Queue()
    try:
        q.get(blocking=True, timeout=0.05)
        assert False, "Should have raised Empty"
    except cthreading.Empty:
        pass


def test_queue_task_done_join():
    q = cthreading.Queue()
    q.put("x")
    q.put("y")
    results: list[str] = []

    def worker():
        while True:
            try:
                item = q.get_nowait()
            except cthreading.Empty:
                break
            results.append(item)
            q.task_done()

    t = stdlib_threading.Thread(target=worker)
    t.start()
    t.join()
    q.join()
    assert sorted(results) == ["x", "y"]


def test_queue_task_done_too_many():
    q = cthreading.Queue()
    try:
        q.task_done()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── LifoQueue ─────────────────────────────────────────────────────

def test_lifo_queue_order():
    lq = cthreading.LifoQueue()
    lq.put(1)
    lq.put(2)
    lq.put(3)
    assert lq.get() == 3
    assert lq.get() == 2
    assert lq.get() == 1


def test_lifo_queue_bounded():
    lq = cthreading.LifoQueue(maxsize=2)
    lq.put("a")
    lq.put("b")
    assert lq.full()
    try:
        lq.put_nowait("c")
        assert False, "Should raise Full"
    except cthreading.Full:
        pass


def test_lifo_queue_task_done_join():
    lq = cthreading.LifoQueue()
    lq.put("x")
    lq.get()
    lq.task_done()
    lq.join()


# ── SimpleQueue ───────────────────────────────────────────────────

def test_simple_queue_basic():
    sq = cthreading.SimpleQueue()
    sq.put("hello")
    sq.put("world")
    assert sq.get() == "hello"
    assert sq.get() == "world"
    assert sq.empty()


def test_simple_queue_empty_exception():
    sq = cthreading.SimpleQueue()
    try:
        sq.get_nowait()
        assert False, "Should raise Empty"
    except cthreading.Empty:
        pass


def test_simple_queue_timeout():
    sq = cthreading.SimpleQueue()
    try:
        sq.get(blocking=True, timeout=0.05)
        assert False, "Should raise Empty"
    except cthreading.Empty:
        pass


# ── PriorityQueue enhancements ───────────────────────────────────

def test_priority_queue_full():
    pq = cthreading.PriorityQueue(maxsize=2)
    pq.put("a", priority=1)
    pq.put("b", priority=2)
    assert pq.full()


def test_priority_queue_nowait():
    pq = cthreading.PriorityQueue()
    pq.put_nowait("item", priority=5)
    assert pq.get_nowait() == "item"


def test_priority_queue_empty_exception():
    pq = cthreading.PriorityQueue()
    try:
        pq.get_nowait()
        assert False, "Should raise Empty"
    except cthreading.Empty:
        pass


# ── Thread ────────────────────────────────────────────────────────

def test_thread_basic():
    result = []
    t = cthreading.Thread(target=lambda: result.append(42))
    assert not t.is_alive()
    t.start()
    t.join()
    assert not t.is_alive()
    assert result == [42]


def test_thread_name():
    t = cthreading.Thread(name="MyThread")
    assert t.name == "MyThread"
    t.name = "Renamed"
    assert t.name == "Renamed"


def test_thread_daemon():
    t = cthreading.Thread(daemon=True)
    assert t.daemon is True


def test_thread_with_args():
    result = []
    t = cthreading.Thread(target=lambda x, y: result.append(x + y), args=(3, 4))
    t.start()
    t.join()
    assert result == [7]


def test_thread_with_kwargs():
    result: list[int] = []
    t = cthreading.Thread(
        target=lambda a=0, b=0: result.append(a * b),
        kwargs={"a": 5, "b": 6},
    )
    t.start()
    t.join()
    assert result == [30]


def test_thread_join_timeout():
    import time as _time
    t = cthreading.Thread(target=lambda: _time.sleep(5))
    t.start()
    t.join(timeout=0.05)
    assert t.is_alive()  # should still be alive (didn't finish)
    # Don't wait for the full 5s — just let it be


# ── Timer ─────────────────────────────────────────────────────────

def test_timer_basic():
    result: list[str] = []
    timer = cthreading.Timer(0.05, lambda: result.append("fired"))
    timer.start()
    timer.join()
    assert result == ["fired"]


def test_timer_cancel():
    result: list[str] = []
    timer = cthreading.Timer(10.0, lambda: result.append("bad"))
    timer.start()
    timer.cancel()
    timer.join()
    assert result == []


# ── Module-level functions ────────────────────────────────────────

def test_active_count():
    count = cthreading.active_count()
    assert count >= 1


def test_get_ident():
    ident = cthreading.get_ident()
    assert isinstance(ident, int)
    assert ident > 0


def test_get_native_id():
    nid = cthreading.get_native_id()
    assert isinstance(nid, int)
    assert nid > 0


def test_main_thread():
    mt = cthreading.main_thread()
    assert mt is not None
    assert mt.name == "MainThread"


def test_current_thread():
    ct = cthreading.current_thread()
    assert ct is not None


def test_enumerate_threads():
    threads = cthreading.enumerate_threads()
    assert isinstance(threads, list)
    assert len(threads) >= 1


def test_timeout_max():
    assert cthreading.TIMEOUT_MAX > 0
