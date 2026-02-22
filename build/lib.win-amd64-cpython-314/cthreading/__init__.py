"""
cthreading — High-performance C-backed threading for Python.

Usage:
    from cthreading import ThreadPool, Lock, Queue, Thread, auto_thread
    from cthreading.monitoring import Ghost, Counter
"""

from cthreading._threading import auto_run_parallel
from cthreading.auto import auto_threaded
from cthreading.monitoring import Counter, Ghost
from cthreading.pool import (
    TIMEOUT_MAX,
    Thread,
    ThreadPool,
    Timer,
    active_count,
    auto_thread,
    cpu_count,
    current_thread,
    enumerate_threads,
    get_ident,
    get_native_id,
    main_thread,
    parallel_map,
    parallel_starmap,
    physical_cpu_count,
    pool_map,
    pool_starmap,
    stack_size,
)
from cthreading.queue import Empty, Full, LifoQueue, PriorityQueue, Queue, SimpleQueue
from cthreading.sync import (
    Barrier,
    BoundedSemaphore,
    BrokenBarrierError,
    Condition,
    Event,
    Lock,
    RLock,
    Semaphore,
)
from cthreading.tasks import TaskBatch, TaskGroup

__all__ = [
    # Thread / Pool
    "Thread",
    "Timer",
    "ThreadPool",
    "auto_thread",
    "auto_run_parallel",
    "cpu_count",
    "physical_cpu_count",
    "parallel_map",
    "parallel_starmap",
    "pool_map",
    "pool_starmap",
    "active_count",
    "current_thread",
    "main_thread",
    "enumerate_threads",
    "get_ident",
    "get_native_id",
    "stack_size",
    "TIMEOUT_MAX",
    # Sync primitives
    "Lock",
    "RLock",
    "Event",
    "Semaphore",
    "BoundedSemaphore",
    "Condition",
    "Barrier",
    "BrokenBarrierError",
    # Queues
    "Queue",
    "PriorityQueue",
    "LifoQueue",
    "SimpleQueue",
    "Empty",
    "Full",
    # Monitoring
    "Ghost",
    "Counter",
    # Tasks
    "TaskBatch",
    "TaskGroup",
    # Auto-patching
    "auto_threaded",
]
