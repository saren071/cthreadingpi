"""
cthreading.tasks — TaskBatch and TaskGroup.

Thin Python wrapper over the C _tasks extension.
Falls back to pure-Python if C unavailable.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from typing import Any, Callable


class TaskBatch:
    """Accumulates tasks and flushes them to a ThreadPool in batch."""

    def __init__(
        self,
        pool: Any,
        flush_threshold: int = 100,
        priority: int = 0,
        group: int = 0,
    ) -> None:
        self._pool = pool
        self._flush_threshold = flush_threshold
        self._priority = priority
        self._group = group
        self._tasks: list[tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any] | None, int]] = []

    def add(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        priority: int | None = None,
    ) -> None:
        if priority is None:
            priority = self._priority
        if args is None:
            args = ()
        self._tasks.append((fn, args, kwargs, priority))
        if self._flush_threshold > 0 and len(self._tasks) >= self._flush_threshold:
            self.flush()

    def flush(self) -> int:
        count = len(self._tasks)
        for fn, args, kwargs, priority in self._tasks:
            self._pool.submit(fn, args, kwargs, priority=priority, group=self._group)
        self._tasks.clear()
        return count

    def pending(self) -> int:
        return len(self._tasks)


class TaskGroup:
    """Groups related tasks for tracking."""

    _next_id = 1

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.group_id = TaskGroup._next_id
        TaskGroup._next_id += 1
        self._submitted = 0
        self._completed = 0
        self._failed = 0

    def submit(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        priority: int = 0,
    ) -> None:
        self._pool.submit(fn, args, kwargs, priority=priority, group=self.group_id)
        self._submitted += 1

    def stats(self) -> dict[str, int]:
        return {
            "group_id": self.group_id,
            "submitted": self._submitted,
            "completed": self._completed,
            "failed": self._failed,
        }


# ---------------------------------------------------------------------------
# Native C extension loading
# ---------------------------------------------------------------------------

def _try_load_native() -> Any | None:
    base_dir = Path(__file__).resolve().parent
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = base_dir / ("_tasks" + suffix)
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("_tasks", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


_native = _try_load_native()
if _native is not None:
    if hasattr(_native, "TaskBatch"):
        TaskBatch = _native.TaskBatch
    if hasattr(_native, "TaskGroup"):
        TaskGroup = _native.TaskGroup


__all__ = [
    "TaskBatch",
    "TaskGroup",
]
