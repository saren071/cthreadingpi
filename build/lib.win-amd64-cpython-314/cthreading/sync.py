"""
cthreading.sync — Lock, RLock, Event, Semaphore, Condition, Barrier.

Thin Python wrappers that ONLY provide access to the C _sync extension.
No stdlib threading usage — everything is C-backed.
If the C extension is not compiled, stubs raise RuntimeError.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any


def _no_native_error() -> None:
    raise RuntimeError(
        "cthreading.sync requires the compiled C extension (_sync). "
        "Build with: pip install -e ."
    )


# ---------------------------------------------------------------------------
# Stubs — replaced by C extension at the bottom of this module.
# These include full method signatures so type-checkers (Pyright) can
# see the API even though the real impl is in C.
# ---------------------------------------------------------------------------

class Lock:
    """C-backed non-reentrant lock. STUB — real impl in _sync.c."""
    def __init__(self) -> None: _no_native_error()
    def acquire(self, blocking: bool = True, timeout: float = -1.0) -> bool: ...
    def release(self) -> None: ...
    def locked(self) -> bool: ...
    def __enter__(self) -> Lock: ...
    def __exit__(self, *args: Any) -> None: ...
    def stats(self) -> dict[str, int]: ...


class RLock:
    """C-backed reentrant lock. STUB — real impl in _sync.c."""
    def __init__(self) -> None: _no_native_error()
    def acquire(self, blocking: bool = True, timeout: float = -1.0) -> bool: ...
    def release(self) -> None: ...
    def __enter__(self) -> RLock: ...
    def __exit__(self, *args: Any) -> None: ...
    def stats(self) -> dict[str, int]: ...


class Event:
    """C-backed manual-reset event. STUB — real impl in _sync.c."""
    def __init__(self) -> None: _no_native_error()
    def is_set(self) -> bool: ...
    def set(self) -> None: ...
    def clear(self) -> None: ...
    def wait(self, timeout: float | None = None) -> bool: ...


class Semaphore:
    """C-backed counting semaphore. STUB — real impl in _sync.c."""
    def __init__(self, value: int = 1, max_value: int = 0) -> None: _no_native_error()
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool: ...
    def release(self, n: int = 1) -> None: ...
    def __enter__(self) -> Semaphore: ...
    def __exit__(self, *args: Any) -> None: ...
    def stats(self) -> dict[str, int]: ...


class BoundedSemaphore:
    """C-backed bounded semaphore. STUB — real impl in _sync.c."""
    def __init__(self, value: int = 1) -> None: _no_native_error()
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool: ...
    def release(self, n: int = 1) -> None: ...
    def __enter__(self) -> BoundedSemaphore: ...
    def __exit__(self, *args: Any) -> None: ...


class Condition:
    """C-backed condition variable. STUB — real impl in _sync.c."""
    def __init__(self, lock: Lock | RLock | None = None) -> None: _no_native_error()
    def acquire(self, *args: Any, **kwargs: Any) -> bool: ...
    def release(self) -> None: ...
    def wait(self, timeout: float | None = None) -> bool: ...
    def wait_for(self, predicate: Any, timeout: float | None = None) -> bool: ...
    def notify(self, n: int = 1) -> None: ...
    def notify_all(self) -> None: ...
    def __enter__(self) -> Condition: ...
    def __exit__(self, *args: Any) -> None: ...


class Barrier:
    """C-backed N-party barrier. STUB — real impl in _sync.c."""
    parties: int
    n_waiting: int
    broken: bool
    def __init__(self, parties: int, action: Any = None, timeout: float | None = None) -> None: _no_native_error()
    def wait(self, timeout: float | None = None) -> int: ...
    def reset(self) -> None: ...
    def abort(self) -> None: ...


class BrokenBarrierError(RuntimeError):
    """Raised when a barrier is broken."""


# ---------------------------------------------------------------------------
# Native C extension loading
# ---------------------------------------------------------------------------

def _try_load_native() -> Any | None:
    base_dir = Path(__file__).resolve().parent
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = base_dir / ("_sync" + suffix)
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("_sync", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


_native = _try_load_native()
if _native is not None:
    if hasattr(_native, "Lock"):
        Lock = _native.Lock
    if hasattr(_native, "RLock"):
        RLock = _native.RLock
    if hasattr(_native, "Event"):
        Event = _native.Event
    if hasattr(_native, "Semaphore"):
        Semaphore = _native.Semaphore
    if hasattr(_native, "Condition"):
        Condition = _native.Condition
    if hasattr(_native, "BoundedSemaphore"):
        BoundedSemaphore = _native.BoundedSemaphore
    if hasattr(_native, "Barrier"):
        Barrier = _native.Barrier
    if hasattr(_native, "BrokenBarrierError"):
        BrokenBarrierError = _native.BrokenBarrierError


__all__ = [
    "Lock",
    "RLock",
    "Event",
    "Semaphore",
    "BoundedSemaphore",
    "Condition",
    "Barrier",
    "BrokenBarrierError",
]
