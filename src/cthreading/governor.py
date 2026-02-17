# src/cthreading/governor.py

from __future__ import annotations

import sys
from typing import Any, get_args, get_origin, get_type_hints

from .contracts import Atomic, Frozen, Ghost, IntGhost


class Managed(type):
    """
    Replaces annotated fields with their descriptor (Ghost/Atomic/Frozen)
    while keeping everything type-strict.
    """
    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> type:
        raw_annotations: dict[str, Any] = namespace.get("__annotations__", {})
        cls = super().__new__(mcs, name, bases, namespace)

        module = sys.modules.get(getattr(cls, "__module__", ""))
        globalns = module.__dict__ if module is not None else None
        localns = dict(vars(cls))

        resolved = get_type_hints(cls, globalns=globalns, localns=localns, include_extras=True)

        for attr_name in raw_annotations.keys():
            type_hint = resolved.get(attr_name)
            if type_hint is None:
                continue

            origin = get_origin(type_hint)
            target_governor: type[Atomic[Any]] | type[Frozen[Any]] | type[Ghost[Any]] | None = None
            base_type = origin or type_hint

            if origin in (Atomic, Frozen, Ghost):
                target_governor = origin
            else:
                metadata = getattr(type_hint, "__metadata__", None)
                if metadata is not None:
                    for item in metadata:
                        if item in (Atomic, Frozen, Ghost):
                            target_governor = item
                            break

                if metadata is not None:
                    args = get_args(type_hint)
                    if args:
                        base_type = args[0]

            if target_governor is Ghost and base_type is int:
                target_governor = IntGhost

            if target_governor:
                current_value = getattr(cls, attr_name, None)
                descriptor = target_governor(initial_value=current_value)
                setattr(cls, attr_name, descriptor)
                descriptor.__set_name__(cls, attr_name)

        return cls


class OmniBase(metaclass=Managed):
    """
    Base class for all managed objects using Ghost/Atomic/Frozen.
    """
    pass
