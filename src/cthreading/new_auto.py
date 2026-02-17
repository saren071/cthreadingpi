"""
cthreading.auto
~~~~~~~~~~~~~~~

AST-rewriting auto-parallelizer.

Decorate any entry-point function with ``@auto_threaded`` and the
transformer rewrites it — and every pure-Python function it transitively
calls in the same module — before Python executes a single bytecode.

Transformations applied
-----------------------
1. List/set comprehensions        →  pool.map / pool.starmap
       [f(x) for x in items]      →  list(pool.map(f, items))
       [f(a,b) for a,b in items]  →  list(pool.starmap(f, items))

2. for-range loops (subscript assignment)
       for i in range(n): out[i] = f(i)
   →   out[:] = pool.map(f, range(n))

3. for-range accumulator / reduction
       for i in range(n): total += f(i)
   →   total += sum(pool.map(f, range(n)))

4. Independent sequential calls in the same scope
       a = fetch_a()      # g does not read a
       b = fetch_b()
       use(a, b)
   →   __ct_fut_1 = pool.submit(fetch_a)
       __ct_fut_2 = pool.submit(fetch_b)
       a, b = __ct_fut_1.result(), __ct_fut_2.result()
       use(a, b)

5. Transitive rewriting — every pure-Python callable defined in the same
   module file is also rewritten recursively (up to depth 8).

Hard limits (physics, not engineering)
---------------------------------------
* A single opaque call  f(big_n)  cannot be split — the loop inside f
  is invisible until f's source is rewritten, which IS done when f lives
  in the same module.  Third-party C extensions are never rewritten.
* Shared mutable state (appending to the same list from N threads) is
  the caller's responsibility.
* Recursive functions, generators (yield), and async def are left alone.

Usage
-----
    from cthreading import auto_threaded

    @auto_threaded          # one decorator on the entry point
    def main():
        results = [heavy(x) for x in big_list]   # → pool.map
        a = fetch_a()                             # → concurrent
        b = fetch_b()                             # → concurrent
        process(a, b)                             # sequential (b depends on a,b)

    main()

    # Or wrap without modifying the original source:
    auto_threaded.run(my_main)
"""

from __future__ import annotations

import ast
import functools
import inspect
import textwrap
import threading
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# Pool access (lazy, thread-safe)
# ---------------------------------------------------------------------------
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
# Tiny AST helpers
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


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

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
        call: ast.Call = elt  # type: ignore[assignment]
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
                    for a, t in zip(call.args, gen.target.elts, strict=True))):
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
            c: ast.Call = s.value  # type: ignore[assignment]
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
            c = s.value  # type: ignore[assignment]
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
                c: ast.Call = s.value  # type: ignore[assignment]
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
            for tgt, c, orig in batch:
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

    # ---- visit_FunctionDef: apply all transforms + fuse ------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Skip generators
        if any(isinstance(n, (ast.Yield, ast.YieldFrom))
               for n in ast.walk(node)):
            return node
        # First, recurse into nested function bodies
        node = self.generic_visit(node)  # type: ignore[assignment]
        # Then fuse any independent sequential calls at this level
        node.body = self._fuse(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node   # leave async alone


# ---------------------------------------------------------------------------
# Source-level rewriter
# ---------------------------------------------------------------------------

_rewritten_ids: set[int] = set()


def _rewrite(fn: Callable) -> Callable:
    """
    Parse fn's source, run the transformer, recompile, and return a new
    callable with the same globals.  Returns fn unchanged on any failure.
    """
    if id(fn) in _rewritten_ids:
        return fn
    if not inspect.isfunction(fn):
        return fn

    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except Exception:
        return fn

    # Strip decorators so exec doesn't re-apply them
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


def _rewrite_callees(fn: Callable, depth: int = 0, max_depth: int = 8) -> None:
    """
    Transitively rewrite every Python function in fn's module that hasn't
    been rewritten yet.
    """
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
# Public proxy
# ---------------------------------------------------------------------------

class _AutoThreaded:
    """
    Single entry-point decorator / runner for automatic parallelism.

        @auto_threaded
        def main(): ...

        auto_threaded.run(main)       # wrap-and-call without modifying source
        auto_threaded.patch()         # optional: route stdlib Thread → pool
        auto_threaded.unpatch()
        auto_threaded.is_patched()
    """

    def __init__(self) -> None:
        self._patched = False
        self._orig_thread: Any = None

    def __call__(self, fn: F) -> F:
        """Rewrite fn and all callees in the same module."""
        new_fn = _rewrite(fn)
        _rewrite_callees(new_fn)
        return new_fn  # type: ignore[return-value]

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Rewrite and immediately call fn."""
        return self(fn)(*args, **kwargs)

    # ---- optional stdlib patch -------------------------------------------

    def patch(self) -> None:
        """Route ``threading.Thread`` starts through the cthreading pool."""
        if self._patched:
            return
        import threading as _t
        self._orig_thread = _t.Thread
        pool = _get_pool()

        class _PooledThread(_t.Thread):
            def start(self_t) -> None:   # noqa: N805
                pool.submit(self_t.run)

        _t.Thread = _PooledThread  # type: ignore[misc]
        self._patched = True

    def unpatch(self) -> None:
        if not self._patched:
            return
        import threading as _t
        _t.Thread = self._orig_thread  # type: ignore[misc]
        self._patched = False

    def is_patched(self) -> bool:
        return self._patched


auto_threaded = _AutoThreaded()