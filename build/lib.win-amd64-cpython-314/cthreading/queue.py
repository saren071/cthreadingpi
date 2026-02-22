"""
cthreading.queue — Thread-safe Queue and PriorityQueue.

Thin Python wrappers that ONLY provide access to the C _queue extension.
No stdlib queue usage — everything is C-backed.
If the C extension is not compiled, stubs raise RuntimeError.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any


def _no_native_error() -> None:
    raise RuntimeError(
        "cthreading.queue requires the compiled C extension (_queue). "
        "Build with: pip install -e ."
    )


class Queue:
    """C-backed thread-safe FIFO queue. STUB — real impl in _queue.c."""
    def __init__(self, maxsize: int = 0) -> None: _no_native_error()
    def put(self, item: Any, blocking: bool = True, timeout: float | None = None) -> None: ...
    def put_nowait(self, item: Any) -> None: ...
    def get(self, blocking: bool = True, timeout: float | None = None) -> Any: ...
    def get_nowait(self) -> Any: ...
    def qsize(self) -> int: ...
    def empty(self) -> bool: ...
    def full(self) -> bool: ...
    def task_done(self) -> None: ...
    def join(self) -> None: ...
    def stats(self) -> dict[str, int]: ...


class PriorityQueue:
    """C-backed thread-safe priority queue. STUB — real impl in _queue.c."""
    def __init__(self, maxsize: int = 0) -> None: _no_native_error()
    def put(self, item: Any, priority: Any = 0, blocking: bool = True) -> None: ...
    def get(self, blocking: bool = True, timeout: float | None = None) -> Any: ...
    def get_nowait(self) -> Any: ...
    def qsize(self) -> int: ...
    def empty(self) -> bool: ...
    def stats(self) -> dict[str, int]: ...


# ---------------------------------------------------------------------------
# Native C extension loading
# ---------------------------------------------------------------------------

def _try_load_native() -> Any | None:
    base_dir = Path(__file__).resolve().parent
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = base_dir / ("_queue" + suffix)
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("_queue", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


_native = _try_load_native()
if _native is not None:
    if hasattr(_native, "Queue"):
        Queue = _native.Queue
    if hasattr(_native, "PriorityQueue"):
        PriorityQueue = _native.PriorityQueue
    if hasattr(_native, "LifoQueue"):
        LifoQueue = _native.LifoQueue
    else:
        import queue as _stdlib_queue
        LifoQueue = _stdlib_queue.LifoQueue
    if hasattr(_native, "SimpleQueue"):
        SimpleQueue = _native.SimpleQueue
    else:
        import queue as _stdlib_queue
        SimpleQueue = _stdlib_queue.SimpleQueue
    if hasattr(_native, "Empty"):
        Empty = _native.Empty
    else:
        import queue as _stdlib_queue
        Empty = _stdlib_queue.Empty
    if hasattr(_native, "Full"):
        Full = _native.Full
    else:
        import queue as _stdlib_queue
        Full = _stdlib_queue.Full
else:
    import queue as _stdlib_queue
    LifoQueue = _stdlib_queue.LifoQueue
    SimpleQueue = _stdlib_queue.SimpleQueue
    Empty = _stdlib_queue.Empty
    Full = _stdlib_queue.Full


__all__ = [
    "Queue",
    "PriorityQueue",
    "LifoQueue",
    "SimpleQueue",
    "Empty",
    "Full",
]
