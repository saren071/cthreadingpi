"""
cthreading.auto — Automatic monkey-patching of stdlib threading/queue.

Replaces stdlib ``threading`` and ``queue`` primitives with cthreading's
C-backed implementations so that **any** application code using the standard
library automatically benefits from native-OS synchronisation primitives.

Usage as decorator::

    from cthreading import auto_threaded

    @auto_threaded
    def main():
        import threading, queue
        lock = threading.Lock()        # ← C-backed Lock
        q    = queue.Queue()           # ← C-backed Queue
        ...

    if __name__ == "__main__":
        main()

Usage as direct runner::

    from cthreading import auto_threaded

    def main():
        ...

    if __name__ == "__main__":
        auto_threaded.run(main)

Usage as simple patcher (no wrapping)::

    from cthreading import auto_threaded
    auto_threaded.patch()
    # All subsequent threading.Lock() etc. are now C-backed.
"""

from __future__ import annotations

import functools
import importlib
import queue
import threading
from typing import Any, Callable, Iterable, TypeVar

from cthreading.pool import cpu_count

_F = TypeVar("_F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Patch state
# ---------------------------------------------------------------------------

_patched: bool = False
_originals: dict[str, Any] = {}


def auto_run_parallel(fn: Callable[[Any], Any], items: Iterable[Any], num_workers: int = 0) -> list[Any]:
    """Python stub for native auto_run_parallel; falls back to parallel_map."""
    try:
        native_mod = importlib.import_module("cthreading._threading")
        native_fn = native_mod.auto_run_parallel
        return native_fn(fn, items, num_workers=num_workers)
    except Exception:
        from cthreading.pool import parallel_map

        return parallel_map(fn, items, num_workers=num_workers)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_patches() -> None:
    """Replace stdlib threading/queue classes with cthreading equivalents."""
    global _patched
    if _patched:
        return

    from cthreading.queue import (
        LifoQueue as CLifoQueue,
    )
    from cthreading.queue import (
        Queue as CQueue,
    )
    from cthreading.queue import (
        SimpleQueue as CSimpleQueue,
    )
    from cthreading.sync import (
        Barrier,
        BoundedSemaphore,
        Condition,
        Event,
        Lock,
        RLock,
        Semaphore,
    )

    # Save originals so we can restore later
    _originals.update({
        "threading.Lock": threading.Lock,
        "threading.RLock": threading.RLock,
        "threading.Event": threading.Event,
        "threading.Semaphore": threading.Semaphore,
        "threading.BoundedSemaphore": threading.BoundedSemaphore,
        "threading.Condition": threading.Condition,
        "threading.Barrier": threading.Barrier,
        "queue.Queue": queue.Queue,
        "queue.LifoQueue": queue.LifoQueue,
        "queue.SimpleQueue": queue.SimpleQueue,
    })

    # Patch threading module
    threading.Lock = Lock  # type: ignore[misc,assignment]
    threading.RLock = RLock  # type: ignore[misc,assignment]
    threading.Event = Event  # type: ignore[misc,assignment]
    threading.Semaphore = Semaphore  # type: ignore[misc,assignment]
    threading.BoundedSemaphore = BoundedSemaphore  # type: ignore[misc,assignment]
    threading.Condition = Condition  # type: ignore[misc,assignment]
    threading.Barrier = Barrier  # type: ignore[misc,assignment]

    # Patch queue module
    queue.Queue = CQueue  # type: ignore[misc,assignment]
    queue.LifoQueue = CLifoQueue  # type: ignore[misc,assignment]
    queue.SimpleQueue = CSimpleQueue  # type: ignore[misc,assignment]

    _patched = True


def _remove_patches() -> None:
    """Restore original stdlib classes."""
    global _patched
    if not _patched:
        return

    for key, val in _originals.items():
        mod_name, attr = key.rsplit(".", 1)
        mod = threading if mod_name == "threading" else queue
        setattr(mod, attr, val)

    _originals.clear()
    _patched = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class auto_threaded:
    """Decorator / utility that monkey-patches stdlib to use cthreading.

    As a **decorator**, patches are applied when the decorated function is
    first defined, and every subsequent call goes through C-backed primitives::

        @auto_threaded
        def main():
            ...
        main()

    As a **runner**, patches are applied and the function is executed
    immediately::

        auto_threaded.run(main)

    As a **manual patcher** (for library / framework code)::

        auto_threaded.patch()   # activate
        auto_threaded.unpatch() # deactivate
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        _apply_patches()
        self._fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], int):
            n: int = args[0]
            workers_kw = kwargs.pop("workers", None)
            workers = max(1, workers_kw if isinstance(workers_kw, int) else cpu_count())
            chunk = max(1, (n + workers - 1) // workers)

            def _job(_: int) -> Any:
                return self._fn(chunk, *args[1:], **kwargs)

            from cthreading.pool import ThreadPool

            pool = ThreadPool(workers)
            try:
                pool.map(_job, range(workers))
            finally:
                pool.shutdown(True)
            return None

        return self._fn(*args, **kwargs)

    # -- Static helpers ------------------------------------------------

    @staticmethod
    def run(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Patch modules and execute *fn* immediately."""
        _apply_patches()
        if args and isinstance(args[0], int):
            n: int = args[0]
            workers_kw = kwargs.pop("workers", None)
            workers = max(1, workers_kw if isinstance(workers_kw, int) else cpu_count())
            chunk = max(1, (n + workers - 1) // workers)

            def _job(_: int) -> Any:
                return fn(chunk, *args[1:], **kwargs)

            from cthreading.pool import ThreadPool

            pool = ThreadPool(workers)
            try:
                pool.map(_job, range(workers))
            finally:
                pool.shutdown(True)
            return None

        return fn(*args, **kwargs)

    @staticmethod
    def run_parallel(fn: Callable[[Any], Any], items: Iterable[Any], num_workers: int = 0) -> list[Any]:
        """Patch modules and execute *fn* over *items* using cthreading's native pool."""
        _apply_patches()
        return auto_run_parallel(fn, items, num_workers=num_workers)

    @staticmethod
    def patch() -> None:
        """Apply monkey-patches without wrapping a function."""
        _apply_patches()

    @staticmethod
    def unpatch() -> None:
        """Remove monkey-patches and restore original stdlib classes."""
        _remove_patches()

    @staticmethod
    def is_patched() -> bool:
        """Return ``True`` if patches are currently active."""
        return _patched


__all__ = ["auto_threaded"]
