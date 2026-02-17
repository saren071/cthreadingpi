# src/cthreading/contracts.py

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from . import monitoring as _monitoring
from .sync import Lock as _SyncLock

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")

class Ghost(Generic[T]):
    default_value: T | None
    name: str

    def __init__(self, initial_value: T | None = None) -> None:
        self.default_value = initial_value
        self.name = ""
        self._cells: weakref.WeakKeyDictionary[
            object, _monitoring.Ghost | _monitoring.Counter
        ] = weakref.WeakKeyDictionary()
        self._cells_lock = _SyncLock()

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def _get_cell(self, instance: object) -> _monitoring.Ghost | _monitoring.Counter:
        with self._cells_lock:
            cell = self._cells.get(instance)
            if cell is None:
                cell = _monitoring.Ghost(initial=self.default_value)
                self._cells[instance] = cell
            return cell

    def _get_value(self, instance: object) -> T | None:
        cell = self._get_cell(instance)
        return cell.get()

    def _set_value(self, instance: object, value: T | None) -> None:
        cell = self._get_cell(instance)
        cell.set(value)

    @overload
    def __get__(self, instance: None, owner: type) -> Ghost[T]: ...

    @overload
    def __get__(self, instance: object, owner: type) -> T | None: ...

    def __get__(self, instance: object, owner: type) -> T | None | Ghost[T]:
        if instance is None:
            return self
        return self._get_value(instance)

    def __set__(self, instance: object, value: T) -> None:
        self._set_value(instance, value)

    def update(self, instance: object, fn: Callable[[T | None], T | None]) -> T | None:
        cell = self._get_cell(instance)
        if hasattr(cell, "update"):
            return cell.update(fn)
        with cell:
            current = cell.get()
            updated = fn(current)
            cell.set(updated)
            return updated

    def get_stats(self, instance: object) -> dict[str, int]:
        cell = self._get_cell(instance)
        if hasattr(cell, "stats_tuple"):
            a, h, v = cell.stats_tuple()
            return {"accesses": int(a), "heat": int(h), "version": int(v)}
        stats: dict[str, int] = cell.stats()
        return stats

    def get_stats_tuple(self, instance: object) -> tuple[int, int, int]:
        cell = self._get_cell(instance)
        if hasattr(cell, "stats_tuple"):
            a, h, v = cell.stats_tuple()
            return int(a), int(h), int(v)
        stats: dict[str, int] = cell.stats()
        return int(stats.get("accesses", 0)), int(stats.get("heat", 0)), int(stats.get("version", 0))


class IntGhost(Ghost[int]):
    def __init__(self, initial_value: int | None = None) -> None:
        super().__init__(initial_value=initial_value)

    class _Proxy:
        __slots__ = ("_desc", "_inst")

        def __init__(self, desc: IntGhost, inst: object) -> None:
            self._desc = desc
            self._inst = inst

        def __iadd__(self, other: int) -> IntGhost._Proxy:
            self._desc._add(self._inst, other)
            return self

        def __int__(self) -> int:
            value = self._desc._get_value(self._inst)
            return 0 if value is None else int(value)

        def __index__(self) -> int:
            return int(self)

        def __repr__(self) -> str:
            return repr(self._desc._get_value(self._inst))

        def __str__(self) -> str:
            return str(self._desc._get_value(self._inst))

        def __format__(self, format_spec: str) -> str:
            return format(self._desc._get_value(self._inst), format_spec)

        def __eq__(self, other: object) -> bool:
            return self._desc._get_value(self._inst) == other

    def _get_cell(self, instance: object) -> _monitoring.Ghost | _monitoring.Counter:
        with self._cells_lock:
            cell = self._cells.get(instance)
            if cell is None:
                initial = 0 if self.default_value is None else int(self.default_value)
                if hasattr(_monitoring, "Counter"):
                    try:
                        cell = _monitoring.Counter(initial=initial)
                    except TypeError:
                        cell = _monitoring.Counter()
                        if hasattr(cell, "set"):
                            cell.set(initial)
                else:
                    try:
                        cell = _monitoring.Ghost(initial=initial)
                    except TypeError:
                        cell = _monitoring.Ghost()
                        if hasattr(cell, "set"):
                            cell.set(initial)
                self._cells[instance] = cell
            return cell

    def _get_value(self, instance: object) -> int | None:
        return int(self._get_cell(instance).get())

    def _set_value(self, instance: object, value: int | None) -> None:
        self._get_cell(instance).set(0 if value is None else int(value))

    def get(self, instance: object) -> int:
        return int(self._get_cell(instance).get())

    def add(self, instance: object, delta: int) -> int:
        cell = self._get_cell(instance)
        result = cell.add(int(delta))
        if result is not None:
            return int(result)
        return self.get(instance)

    def _add(self, instance: object, delta: int) -> None:
        self._get_cell(instance).add(int(delta))

    @overload
    def __get__(self, instance: None, owner: type) -> IntGhost: ...

    @overload
    def __get__(self, instance: object, owner: type) -> int | None: ...

    def __get__(self, instance: object, owner: type) -> Any:
        if instance is None:
            return self
        self._get_cell(instance)
        return IntGhost._Proxy(self, instance)

    def __set__(self, instance: object, value: int) -> None:
        if isinstance(value, IntGhost._Proxy) and value._desc is self and value._inst is instance:
            return
        self._set_value(instance, value)


class Atomic(Generic[T]):
    default_value: T | None
    name: str

    def __init__(self, initial_value: T | None = None) -> None:
        self.default_value = initial_value
        self.name = ""
        self._cells: weakref.WeakKeyDictionary[object, _monitoring.Ghost] = weakref.WeakKeyDictionary()
        self._cells_lock = _SyncLock()

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def _get_cell(self, instance: object) -> _monitoring.Ghost:
        with self._cells_lock:
            cell = self._cells.get(instance)
            if cell is None:
                cell = _monitoring.Ghost(initial=self.default_value)
                self._cells[instance] = cell
            return cell

    def __get__(self, instance: object, owner: type) -> T | None | Atomic[T]:
        if instance is None:
            return self
        return self._get_cell(instance).get()

    def __set__(self, instance: object, value: T) -> None:
        self._get_cell(instance).set(value)

    def update(self, instance: object, fn: "Callable[[T | None], T | None]") -> T | None:
        cell = self._get_cell(instance)
        if hasattr(cell, "update"):
            return cell.update(fn)
        with cell:
            current = cell.get()
            updated = fn(current)
            cell.set(updated)
            return updated


class Frozen(Generic[T]):
    default_value: T | None
    name: str

    def __init__(self, initial_value: T | None = None) -> None:
        self.default_value = initial_value
        self.name = ""
        self._cells: weakref.WeakKeyDictionary[object, _monitoring.Ghost] = weakref.WeakKeyDictionary()
        self._cells_lock = _SyncLock()
        self._frozen: weakref.WeakSet[object] = weakref.WeakSet()

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def _get_cell(self, instance: object) -> _monitoring.Ghost:
        with self._cells_lock:
            cell = self._cells.get(instance)
            if cell is None:
                cell = _monitoring.Ghost(initial=self.default_value)
                self._cells[instance] = cell
            return cell

    def __get__(self, instance: object, owner: type) -> T | None | Frozen[T]:
        if instance is None:
            return self
        return self._get_cell(instance).get()

    def __set__(self, instance: object, value: T) -> None:
        with self._cells_lock:
            if instance in self._frozen:
                raise PermissionError(f"Modification denied: '{self.name}' is Frozen.")
            self._frozen.add(instance)
        self._get_cell(instance).set(value)
