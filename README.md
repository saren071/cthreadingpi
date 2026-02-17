# cthreadingpi

High-performance C-backed threading for Python 3.14t.

**Version:** 0.1.0  \
**Author:** Sarenian <ozlohu99@gmail.com>  \
**License:** unlicense  \
**Python Version:** >=3.14

## Installation

```bash
pip install cthreadingpi
```

or from source:

```bash
pip install .
```

## Quick start

```python
from cthreading import Thread, ThreadPool, parallel_map

def work(x: int) -> int:
    return x * 2

# Plain thread (mirrors stdlib)
t = Thread(target=work, args=(21,))
t.start(); t.join()

# Pool usage
with ThreadPool(num_workers=4) as pool:
    results = pool.map(work, range(10))

# Functional helper on the default pool
results = parallel_map(work, range(10))
```

## Capabilities (at a glance)

- **C-backed primitives**: `Thread`, `ThreadPool`, `Lock`, `RLock`, `Event`, `Semaphore`, `Barrier`, and `Queue` family.
- **Monitoring**: Ghost cells and sharded Counter with contention/access telemetry.
- **Auto-threaded**: experimental AST-based parallelization and stdlib patching (**currently broken and being fixed**).

## Threads and pools

- Run over data: `parallel_map(fn, items, num_workers=0)`, `parallel_starmap(fn, items_of_tuples, num_workers=0)`.
- Own pool: `with ThreadPool(num_workers=4) as pool: pool.map(...); pool.starmap(...); pool.submit(fn, args=(), kwargs={}, priority=0, group=0); pool.shutdown(wait=True); pool.stats()`.
- Decorate to always use the default pool: `@auto_thread` (from `cthreading.pool`).
- Inspect threads: `active_count()`, `enumerate_threads()`, `current_thread()`, `main_thread()`, `get_ident()`, `get_native_id()`, `stack_size()`, `TIMEOUT_MAX`; `Timer` if the C extension exposes it.

## Synchronization

- Locks: `Lock`, `RLock` (`acquire(blocking=True, timeout=-1.0)`, `release()`, `locked()`, context managers, `stats()` when available).
- Events: `Event.set()`, `clear()`, `is_set()`, `wait(timeout=None)`.
- Semaphores: `Semaphore(value=1, max_value=0)`, `BoundedSemaphore(value=1)` with `acquire`/`release`, context managers.
- Conditions: `Condition(lock=None)` with `wait`, `wait_for`, `notify`, `notify_all`.
- Barriers: `Barrier(parties, action=None, timeout=None)` with `wait`, `reset`, `abort`, `BrokenBarrierError` on failure.

## Queues

- FIFO: `Queue(maxsize=0)` with `put`, `put_nowait`, `get`, `get_nowait`, `qsize`, `empty`, `full`, `task_done`, `join`, `stats`.
- Priority: `PriorityQueue(maxsize=0)` with `put(item, priority=0, blocking=True)`, `get`, `get_nowait`, `qsize`, `empty`, `stats`.
- Others: `LifoQueue`, `SimpleQueue`, exceptions `Empty`, `Full`.

## Monitoring: Ghost and Counter

Telemetry toggle (off by default):

```python
from cthreading.monitoring import set_enabled

set_enabled(True)   # count accesses/heat/version
set_enabled(False)  # stop counting
```

Ghost cell (contention-aware lockable cell):

```python
from cthreading.monitoring import Ghost, set_enabled

set_enabled(True)
g = Ghost(initial=0)
g.add(5)
with g:
    g.set(g.get() + 1)
print(g.get())
print(g.stats())
```

Ghost as descriptor (per-instance field) and sharded int counter:

```python
from cthreading.contracts import Ghost, IntGhost
from cthreading.monitoring import set_enabled

set_enabled(True)

class Worker:
    state: str | None = Ghost(initial_value=None)
    count: IntGhost = IntGhost(0)

w = Worker()
w.state = "ready"
w.count += 1
print(type(w).__dict__["count"].get_stats(w))
```

Sharded Counter (integer-only):

```python
from cthreading.monitoring import Counter

c = Counter(initial=0, shards=64)
for _ in range(100):
    c.add(1)
print(c.get())
print(c.stats())
```

## Auto-threaded (experimental / broken)

`auto_threaded` rewrites certain pure patterns to run in the thread pool and can monkey-patch stdlib primitives. It is **currently broken and under active repair**. Use explicit pools instead for now.

Illustrative (may not work until fixed):

```python
from cthreading import auto_threaded

@auto_threaded
def main(items: list[int]) -> list[int]:
    doubled = [abs(x) for x in items]
    a = expensive(items[0]); b = expensive(items[1])
    return [a, b, sum(doubled)]

if __name__ == "__main__":
    print(main(list(range(10))))

# Optional global patch (experimental)
auto_threaded.patch()
# ... run app ...
auto_threaded.unpatch()
```
