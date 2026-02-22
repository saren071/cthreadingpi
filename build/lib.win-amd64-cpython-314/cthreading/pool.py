"""
cthreading.pool — ThreadPool, auto_thread decorator, cpu_count.

Thin Python wrappers that ONLY provide access to the C _threading extension.
No stdlib threading or concurrent.futures usage — everything is C-backed.
If the C extension is not compiled, stubs raise RuntimeError.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable, Iterable


def _no_native_error() -> None:
    raise RuntimeError(
        "cthreading requires the compiled C extension (_threading). "
        "Build with: pip install -e ."
    )


def cpu_count() -> int:
    """Return the number of CPU cores (logical)."""
    return os.cpu_count() or 1


def physical_cpu_count() -> int:
    """Return the number of physical CPU cores (ignoring hyper-threading)."""
    try:
        n = os.cpu_count() or 1
        return max(1, (n + 1) // 2)
    except Exception:
        return 1


# All pool/map/thread functions are provided by the C extension.
# These stubs exist only so the names are importable before the
# extension is loaded at the bottom of this module.

def parallel_map(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    num_workers: int = 0,
) -> list[Any]:
    """Stub — replaced by C extension at import time."""
    _no_native_error()
    return []  # unreachable


def parallel_starmap(
    fn: Callable[..., Any],
    items: Iterable[tuple[Any, ...]],
    num_workers: int = 0,
) -> list[Any]:
    """Stub — replaced by C extension at import time."""
    _no_native_error()
    return []  # unreachable


def pool_map(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    num_workers: int = 0,
) -> list[Any]:
    """Stub — replaced by C extension at import time."""
    _no_native_error()
    return []  # unreachable


def pool_starmap(
    fn: Callable[..., Any],
    items: Iterable[tuple[Any, ...]],
    num_workers: int = 0,
) -> list[Any]:
    """Stub — replaced by C extension at import time."""
    _no_native_error()
    return []  # unreachable


class ThreadPool:
    """C-backed thread pool. STUB — real impl in _threading.c."""
    def __init__(self, num_workers: int = 0) -> None: _no_native_error()
    @property
    def num_workers(self) -> int: ...
    def submit(self, fn: Callable[..., Any], args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None, priority: int = 0, group: int = 0) -> None: ...
    def wrap(self, fn: Callable[..., Any], args: tuple[Any, ...] | None = None) -> None: ...
    def map(self, fn: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]: ...
    def starmap(self, fn: Callable[..., Any], items: Iterable[tuple[Any, ...]]) -> list[Any]: ...
    def shutdown(self, wait: bool = True) -> None: ...
    def stats(self) -> dict[str, Any]: ...


class auto_thread:
    """Decorator: calls are submitted to the C-backed default pool. STUB — real impl in _threading.c."""
    def __init__(self, fn: Callable[..., Any]) -> None: _no_native_error()
    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


def set_default_pool_size(n: int) -> None:  # pragma: no cover
    """Stub — replaced by C extension at import time."""
    _no_native_error()


def get_default_pool() -> ThreadPool:  # pragma: no cover
    """Stub — replaced by C extension at import time."""
    _no_native_error()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Native C extension loading
# ---------------------------------------------------------------------------

def _try_load_native() -> Any | None:
    base_dir = Path(__file__).resolve().parent
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = base_dir / ("_threading" + suffix)
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("_threading", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


_native = _try_load_native()
if _native is not None:
    if hasattr(_native, "ThreadPool"):
        ThreadPool = _native.ThreadPool
    if hasattr(_native, "auto_thread"):
        auto_thread = _native.auto_thread
    if hasattr(_native, "cpu_count"):
        cpu_count = _native.cpu_count
    if hasattr(_native, "physical_cpu_count"):
        physical_cpu_count = _native.physical_cpu_count
    if hasattr(_native, "set_default_pool_size"):
        set_default_pool_size = _native.set_default_pool_size
    if hasattr(_native, "get_default_pool"):
        get_default_pool = _native.get_default_pool
    if hasattr(_native, "parallel_map"):
        parallel_map = _native.parallel_map
    if hasattr(_native, "parallel_starmap"):
        parallel_starmap = _native.parallel_starmap
    if hasattr(_native, "pool_map"):
        pool_map = _native.pool_map
    if hasattr(_native, "pool_starmap"):
        pool_starmap = _native.pool_starmap
    if hasattr(_native, "Thread"):
        Thread = _native.Thread
    if hasattr(_native, "Timer"):
        Timer = _native.Timer
    if hasattr(_native, "active_count"):
        active_count = _native.active_count
    if hasattr(_native, "current_thread"):
        current_thread = _native.current_thread
    if hasattr(_native, "main_thread"):
        main_thread = _native.main_thread
    if hasattr(_native, "enumerate"):
        enumerate_threads = _native.enumerate
    if hasattr(_native, "get_ident"):
        get_ident = _native.get_ident
    if hasattr(_native, "get_native_id"):
        get_native_id = _native.get_native_id
    if hasattr(_native, "stack_size"):
        stack_size = _native.stack_size
    if hasattr(_native, "TIMEOUT_MAX"):
        TIMEOUT_MAX = _native.TIMEOUT_MAX


__all__ = [
    "ThreadPool",
    "auto_thread",
    "cpu_count",
    "physical_cpu_count",
    "parallel_map",
    "parallel_starmap",
    "pool_map",
    "pool_starmap",
    "set_default_pool_size",
    "get_default_pool",
    "Thread",
    "Timer",
    "active_count",
    "current_thread",
    "main_thread",
    "enumerate_threads",
    "get_ident",
    "get_native_id",
    "stack_size",
    "TIMEOUT_MAX",
]
