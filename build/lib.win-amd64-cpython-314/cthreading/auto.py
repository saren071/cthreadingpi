"""
cthreading.auto — Automatic Parallelism & Native Primitives.

This module consolidates two major features:
1. **AST-Rewriting (The "New" Auto):** Automatically parallelizes comprehensions,
   for-loops, and sequential function calls by rewriting Python bytecode at runtime.
2. **Native Monkey-Patching (The "Old" Auto):** Replaces standard library
   ``threading`` and ``queue`` primitives with cthreading's C-backed implementations
   (Lock, Event, Queue, etc.) for OS-native synchronization.

Usage
-----
    from cthreading import auto_threaded

    # 1. AST Rewriting (Auto-Parallelism)
    @auto_threaded
    def main():
        results = [heavy(x) for x in big_list]   # → automatically runs in pool
        process(results)

    # 2. Global Patching (C-backed Primitives)
    auto_threaded.patch()  # Replaces threading.Lock, queue.Queue, threading.Thread, etc.
    # ... application code ...
    auto_threaded.unpatch()
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import queue
import textwrap
import threading
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

# ---------------------------------------------------------------------------
# Types and Globals
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])

_patched: bool = False
_originals: dict[str, Any] = {}

# Pool access (lazy, thread-safe)
_pool_singleton: Any = None
_pool_lock = threading.Lock()

def _get_pool() -> Any:
    global _pool_singleton
    if _pool_singleton is None:
        with _pool_lock:
            if _pool_singleton is None:
                from cthreading._threading import get_default_pool
                _pool_singleton = get_default_pool()
    return _pool_singleton

# ---------------------------------------------------------------------------
# 1. Native Primitives Patching (From Old auto.py)
# ---------------------------------------------------------------------------

def _patch_primitives() -> None:
    """Replace stdlib threading/queue classes with cthreading equivalents."""
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

    # Save originals
    if not _originals:
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
            # Thread is handled separately in the class, but we track it here implies logic separation
        })

    # Patch threading module
    threading.Lock = Lock
    threading.RLock = RLock
    threading.Event = Event
    threading.Semaphore = Semaphore
    threading.BoundedSemaphore = BoundedSemaphore
    threading.Condition = Condition
    threading.Barrier = Barrier

    # Patch queue module
    queue.Queue = CQueue
    queue.LifoQueue = CLifoQueue
    queue.SimpleQueue = CSimpleQueue


def _unpatch_primitives() -> None:
    """Restore original stdlib classes."""
    for key, val in _originals.items():
        if key == "threading.Thread":
            continue # Handled by class logic
        mod_name, attr = key.rsplit(".", 1)
        mod = threading if mod_name == "threading" else queue
        setattr(mod, attr, val)


def _auto_run_parallel(fn: Callable[[Any], Any], items: Iterable[Any], num_workers: int = 0) -> list[Any]:
    """Helper to run a function in parallel over items."""
    try:
        native_mod = importlib.import_module("cthreading._threading")
        return native_mod.auto_run_parallel(fn, items, num_workers=num_workers)
    except Exception:
        from cthreading.pool import parallel_map
        return parallel_map(fn, items, num_workers=num_workers)


# ---------------------------------------------------------------------------
# 2. AST Rewriting Logic (From New auto_.py)
# ---------------------------------------------------------------------------

_POOL_VAR   = "__ct_pool__"
_FUT_PREFIX = "__ct_fut_"

def _names_read(node: ast.AST) -> set[str]:
    return {
        n.id for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }

def _is_pure_call(node: ast.AST) -> bool:
    """True iff node is a Call with no assignments/yield/await inside."""
    if not isinstance(node, ast.Call):
        return False
    return not any(
        isinstance(c, (ast.Assign, ast.AugAssign, ast.AnnAssign,
                       ast.Yield, ast.YieldFrom, ast.Await))
        for c in ast.walk(node)
    )

def _pool_attr(attr: str) -> ast.Attribute:
    return ast.Attribute(
        value=ast.Name(id=_POOL_VAR, ctx=ast.Load()),
        attr=attr, ctx=ast.Load(),
    )

def _call(func: ast.expr, *args: ast.expr,
          kws: list[ast.keyword] | None = None) -> ast.Call:
    return ast.Call(func=func, args=list(args), keywords=kws or [])

class _Transformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._n = 0

    def _fut(self) -> str:
        self._n += 1
        return f"{_FUT_PREFIX}{self._n}"

    # ---- 1. comprehensions ------------------------------------------------
    def _map_comp(self, elt: ast.expr,
                  gen: ast.comprehension) -> ast.Call | None:
        if gen.ifs or not _is_pure_call(elt):
            return None
        call: ast.Call = elt
        
        # Simple: [f(x) for x in items]
        if (isinstance(gen.target, ast.Name) and
                len(call.args) == 1 and not call.keywords and
                isinstance(call.args[0], ast.Name) and
                call.args[0].id == gen.target.id):
            return _call(_pool_attr("map"), call.func, gen.iter)
        
        # Starmap: [f(a,b) for a,b in items]
        if (isinstance(gen.target, ast.Tuple) and
                not call.keywords and
                len(call.args) == len(gen.target.elts) and
                all(isinstance(a, ast.Name) and isinstance(t, ast.Name)
                    and a.id == t.id
                    for a, t in zip(call.args, gen.target.elts, strict=False))):
            return _call(_pool_attr("starmap"), call.func, gen.iter)
        return None

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        self.generic_visit(node)
        if len(node.generators) == 1:
            m = self._map_comp(node.elt, node.generators[0])
            if m:
                return _call(ast.Name(id="list", ctx=ast.Load()), m)
        return node

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        self.generic_visit(node)
        if len(node.generators) == 1:
            m = self._map_comp(node.elt, node.generators[0])
            if m:
                return _call(ast.Name(id="set", ctx=ast.Load()), m)
        return node

    # ---- 2 & 3. for-range loops ------------------------------------------
    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)
        if node.orelse:
            return node
        if not (isinstance(node.iter, ast.Call) and
                isinstance(node.iter.func, ast.Name) and
                node.iter.func.id == "range" and
                isinstance(node.target, ast.Name)):
            return node
        loop_var = node.target.id
        body = node.body
        if len(body) != 1:
            return node
        s = body[0]

        # Reduction: total += f(i)
        if (isinstance(s, ast.AugAssign) and
                isinstance(s.op, ast.Add) and
                isinstance(s.target, ast.Name) and
                _is_pure_call(s.value)):
            c: ast.Call = s.value
            if (len(c.args) == 1 and not c.keywords and
                    isinstance(c.args[0], ast.Name) and
                    c.args[0].id == loop_var and
                    _names_read(s.value) <= {loop_var}):
                map_c = _call(_pool_attr("map"), c.func, node.iter)
                sum_c = _call(ast.Name(id="sum", ctx=ast.Load()), map_c)
                repl = ast.AugAssign(
                    target=s.target, op=ast.Add(), value=sum_c,
                    lineno=node.lineno, col_offset=node.col_offset,
                )
                return ast.fix_missing_locations(repl)

        # Subscript assignment: out[i] = f(i)
        if (isinstance(s, ast.Assign) and
                len(s.targets) == 1 and
                isinstance(s.targets[0], ast.Subscript) and
                isinstance(s.targets[0].slice, ast.Name) and
                s.targets[0].slice.id == loop_var and
                _is_pure_call(s.value)):
            c: ast.Call = s.value
            if (len(c.args) == 1 and not c.keywords and
                    isinstance(c.args[0], ast.Name) and
                    c.args[0].id == loop_var):
                container = s.targets[0].value
                repl = ast.Assign(
                    targets=[ast.Subscript(
                        value=container, slice=ast.Slice(), ctx=ast.Store()
                    )],
                    value=_call(_pool_attr("map"), c.func, node.iter),
                    lineno=node.lineno, col_offset=node.col_offset,
                )
                return ast.fix_missing_locations(repl)
        return node

    # ---- 4. independent call fusion --------------------------------------
    def _fuse(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        i = 0
        while i < len(stmts):
            batch: list[tuple[str, ast.Call, ast.stmt]] = []
            written: set[str] = set()
            j = i
            while j < len(stmts):
                s = stmts[j]
                if not (isinstance(s, ast.Assign) and
                        len(s.targets) == 1 and
                        isinstance(s.targets[0], ast.Name) and
                        _is_pure_call(s.value)):
                    break
                c: ast.Call = s.value
                if _names_read(c) & written:
                    break
                tgt = s.targets[0].id
                batch.append((tgt, c, s))
                written.add(tgt)
                j += 1

            if len(batch) < 2:
                out.append(stmts[i])
                i += 1
                continue

            # submit phase
            futs: list[str] = []
            for _tgt, c, orig in batch:
                fn = self._fut()
                futs.append(fn)
                sub = ast.Assign(
                    targets=[ast.Name(id=fn, ctx=ast.Store())],
                    value=_call(_pool_attr("submit"),
                                c.func, *c.args, kws=c.keywords),
                    lineno=orig.lineno, col_offset=orig.col_offset,
                )
                out.append(ast.fix_missing_locations(sub))

            # gather phase: a, b = fut1.result(), fut2.result()
            res_calls = [
                _call(ast.Attribute(
                    value=ast.Name(id=fn, ctx=ast.Load()),
                    attr="result", ctx=ast.Load()))
                for fn in futs
            ]
            gather = ast.Assign(
                targets=[ast.Tuple(
                    elts=[ast.Name(id=t, ctx=ast.Store()) for t, _, _ in batch],
                    ctx=ast.Store(),
                )],
                value=ast.Tuple(elts=res_calls, ctx=ast.Load()),
                lineno=batch[-1][2].lineno, col_offset=0,
            )
            out.append(ast.fix_missing_locations(gather))
            i = j
        return out

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Skip generators
        if any(isinstance(n, (ast.Yield, ast.YieldFrom))
               for n in ast.walk(node)):
            return node
        node = self.generic_visit(node)
        node.body = self._fuse(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node   # leave async alone

_rewritten_ids: set[int] = set()

def _rewrite(fn: Callable[..., Any]) -> Callable[..., Any]:
    if id(fn) in _rewritten_ids:
        return fn
    if not inspect.isfunction(fn):
        return fn

    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except Exception:
        return fn

    # Strip decorators
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []

    new_tree = _Transformer().visit(tree)
    ast.fix_missing_locations(new_tree)

    globs: dict[str, Any] = {**fn.__globals__, _POOL_VAR: _get_pool()}

    try:
        code  = compile(new_tree, inspect.getfile(fn), "exec")
        local: dict[str, Any] = {}
        exec(code, globs, local)   # noqa: S102
    except Exception:
        return fn

    new_fn = local.get(fn.__name__)
    if not callable(new_fn):
        return fn

    functools.update_wrapper(new_fn, fn)
    new_fn.__globals__[_POOL_VAR] = _get_pool()

    _rewritten_ids.add(id(fn))
    _rewritten_ids.add(id(new_fn))
    return new_fn

def _rewrite_callees(fn: Callable[..., Any], depth: int = 0, max_depth: int = 8) -> None:
    if depth >= max_depth:
        return
    mod_file = getattr(inspect.getmodule(fn), "__file__", None)
    if not mod_file:
        return
    globs = getattr(fn, "__globals__", {})
    for name, obj in list(globs.items()):
        if (inspect.isfunction(obj) and
                id(obj) not in _rewritten_ids and
                getattr(inspect.getmodule(obj), "__file__", None) == mod_file):
            new_obj = _rewrite(obj)
            if new_obj is not obj:
                globs[name] = new_obj
                _rewrite_callees(new_obj, depth + 1, max_depth)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class _AutoThreaded:
    """
    Consolidated decorator and patch manager.
    
    As a decorator:
        @auto_threaded
        def main(): ...
    
    As a patcher:
        auto_threaded.patch()
    """

    def __init__(self) -> None:
        self._patched = False
        self._orig_thread: Any = None

    def __call__(self, fn: F) -> F:
        """Rewrite fn and all callees in the same module."""
        new_fn: F = _rewrite(fn)
        _rewrite_callees(new_fn)
        return new_fn

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Rewrite and immediately call fn."""
        return self(fn)(*args, **kwargs)

    @staticmethod
    def run_parallel(fn: Callable[[Any], Any], items: Iterable[Any], num_workers: int = 0) -> list[Any]:
        """Manually execute *fn* over *items* using cthreading's native pool."""
        return _auto_run_parallel(fn, items, num_workers=num_workers)

    # ---- Global Patching (Thread + Primitives) ---------------------------

    def patch(self) -> None:
        """
        Apply global patches to stdlib:
        1. threading.Lock/Event/etc -> cthreading native equivalents.
        2. threading.Thread -> routed to cthreading Pool.
        """
        if self._patched:
            return

        # 1. Patch Primitives (Lock, Queue, etc) - From old auto.py
        _patch_primitives()

        # 2. Patch Threading.Thread - From new auto_.py
        import threading as _t
        self._orig_thread = _t.Thread
        pool = _get_pool()

        class _PooledThread(_t.Thread):
            def start(self) -> None:
                pool.submit(self.run)

        _t.Thread = _PooledThread
        
        # Mark as patched
        global _patched
        self._patched = True
        _patched = True

    def unpatch(self) -> None:
        """Remove monkey-patches and restore original stdlib classes."""
        if not self._patched:
            return

        # 1. Unpatch Primitives
        _unpatch_primitives()

        # 2. Unpatch Threading.Thread
        import threading as _t
        if self._orig_thread:
            _t.Thread = self._orig_thread

        global _patched
        self._patched = False
        _patched = False
        _originals.clear()

    def is_patched(self) -> bool:
        return self._patched

auto_threaded = _AutoThreaded()

__all__ = ["auto_threaded"]