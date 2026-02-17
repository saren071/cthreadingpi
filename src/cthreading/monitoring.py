"""
cthreading.monitoring — Ghost cells, sharded counters, telemetry.

Thin Python wrapper over the C _monitoring extension.
Falls back to pure-Python implementations if the C extension is unavailable.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .sync import Lock as _Lock

if TYPE_CHECKING:
    from types import TracebackType

_enabled = False


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = bool(enabled)
    if _native is not None and hasattr(_native, "set_enabled"):
        _native.set_enabled(_enabled)


def enabled() -> bool:
    if _native is not None and hasattr(_native, "enabled"):
        return bool(_native.enabled())
    return _enabled


class Ghost:
    """A re-entrant locked cell for any Python object with contention telemetry."""

    def __init__(self, initial: Any = None) -> None:
        self._lock = _Lock()
        self._value: Any = initial
        self._accesses = 0
        self._heat = 0
        self._version = 0

    def __enter__(self) -> Ghost:
        if not self._lock.acquire(blocking=False):
            if enabled():
                self._heat += 1
            self._lock.acquire()
        if enabled():
            self._accesses += 1
            self._version += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._lock.release()

    def stats(self) -> dict[str, int]:
        return {"accesses": self._accesses, "heat": self._heat, "version": self._version}

    def stats_tuple(self) -> tuple[int, int, int]:
        return self._accesses, self._heat, self._version

    def get(self) -> Any:
        with self:
            return self._value

    def set(self, value: Any) -> None:
        with self:
            self._value = value

    def add(self, delta: Any) -> Any:
        with self:
            current = self._value
            if current is None:
                current = 0
            self._value = current + delta
            return self._value

    def update(self, fn: Any) -> Any:
        with self:
            self._value = fn(self._value)
            return self._value


class Counter:
    """A sharded contention-aware int counter."""

    def __init__(self, initial: int = 0, shards: int = 64) -> None:
        if shards <= 0:
            raise ValueError("shards must be > 0")
        self._locks = [_Lock() for _ in range(shards)]
        self._values = [0] * shards
        self._accesses = [0] * shards
        self._heat = [0] * shards
        self._version = [0] * shards
        self._values[0] = int(initial)

    def _pick(self) -> int:
        import _thread
        return _thread.get_ident() % len(self._locks)

    def add(self, delta: int) -> None:
        idx = self._pick()
        lock = self._locks[idx]
        if not lock.acquire(blocking=False):
            if enabled():
                self._heat[idx] += 1
            lock.acquire()
        try:
            if enabled():
                self._accesses[idx] += 1
                self._version[idx] += 1
            self._values[idx] += int(delta)
        finally:
            lock.release()

    def __enter__(self) -> Counter:
        for i, lock in enumerate(self._locks):
            if not lock.acquire(blocking=False):
                if enabled():
                    self._heat[i] += 1
                lock.acquire()
            if enabled():
                self._accesses[i] += 1
                self._version[i] += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for lock in reversed(self._locks):
            lock.release()

    def get(self) -> int:
        return int(sum(self._values))

    def set(self, value: int) -> None:
        for lock in self._locks:
            lock.acquire()
        try:
            for i in range(len(self._values)):
                self._values[i] = 0
            self._values[0] = int(value)
        finally:
            for lock in reversed(self._locks):
                lock.release()

    def stats_tuple(self) -> tuple[int, int, int]:
        return int(sum(self._accesses)), int(sum(self._heat)), int(sum(self._version))

    def stats(self) -> dict[str, int]:
        a, h, v = self.stats_tuple()
        return {"accesses": a, "heat": h, "version": v}


# ---------------------------------------------------------------------------
# Native C extension loading
# ---------------------------------------------------------------------------

def _try_load_native() -> Any | None:
    base_dir = Path(__file__).resolve().parent
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = base_dir / ("_monitoring" + suffix)
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("_monitoring", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


_native = _try_load_native()
if _native is not None and hasattr(_native, "Ghost"):
    Ghost = _native.Ghost

if _native is not None and hasattr(_native, "Counter"):
    Counter = _native.Counter


__all__ = [
    "Ghost",
    "Counter",
    "set_enabled",
    "enabled",
]
