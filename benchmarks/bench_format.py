"""
bench_format.py — Shared benchmark formatting for cthreading benchmark suite.
===============================================================================

All benchmarks import from here for consistent output, metrics collection,
and JSON export for aggregation by run_all_benchmarks.py.

Every benchmark in the suite uses the SAME visual style: the head-to-head
comparison layout with section bars, speedup tags, gate pass/fail badges,
delta percentages, and a per-suite + final aggregate summary table.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

MIN_SPEEDUP = 15.0  # gate: cthreading must be >= this vs stdlib

try:
    from cthreading import physical_cpu_count
    PHYSICAL = physical_cpu_count()
except Exception:
    PHYSICAL = max((os.cpu_count() or 4) // 2, 1)

CPUS = os.cpu_count() or 4
WORKERS = PHYSICAL
FREE_THREADED = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()

BAR = "\u2501" * 100          # ━
THIN = "\u2500" * 84          # ─
W = BAR

# ═══════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Metrics:
    """Unified result container for every benchmark in the suite."""
    label: str                          # e.g. "stdlib", "cthreading", "sequential"
    ops: int = 0                        # total logical operations
    elapsed: float = 0.0                # wall-clock seconds
    correct: bool = True
    final_value: Any = None             # verification value (counter, checksum, …)
    seq_elapsed: float = 0.0            # sequential baseline time (0 → no baseline)
    latencies_ns: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ops_per_sec(self) -> float:
        return self.ops / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def ns_per_op(self) -> float:
        return (self.elapsed * 1e9) / self.ops if self.ops > 0 else 0.0

    @property
    def speedup_vs_seq(self) -> float:
        if self.elapsed <= 0 or self.seq_elapsed <= 0:
            return 0.0
        return self.seq_elapsed / self.elapsed

    @property
    def parallel_eff(self) -> float:
        return self.speedup_vs_seq / WORKERS * 100 if WORKERS > 0 else 0.0

    def latency_stats(self) -> dict[str, float]:
        if not self.latencies_ns:
            return {}
        import statistics as _st
        s = sorted(self.latencies_ns)
        n = len(s)
        return {
            "min_ns": s[0], "max_ns": s[-1],
            "mean_ns": _st.mean(s), "median_ns": _st.median(s),
            "p95_ns": s[int(n * 0.95)], "p99_ns": s[int(n * 0.99)],
            "stdev_ns": _st.stdev(s) if n > 1 else 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "ops": self.ops,
            "elapsed": self.elapsed,
            "correct": self.correct,
            "final_value": self.final_value,
            "seq_elapsed": self.seq_elapsed,
            "ops_per_sec": self.ops_per_sec,
            "ns_per_op": self.ns_per_op,
            "speedup_vs_seq": self.speedup_vs_seq,
            "parallel_eff": self.parallel_eff,
            "extra": self.extra,
        }


# Global collector — benchmarks append here; export_json serialises it.
ALL_RESULTS: list[tuple[str, str, Metrics, Metrics]] = []


# ═══════════════════════════════════════════════════════════════════
# FORMATTING HELPERS
# ═══════════════════════════════════════════════════════════════════

def fmt_num(n: float) -> str:
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.1f}"


def fmt_ns(ns: float) -> str:
    if ns >= 1e6:
        return f"{ns / 1e6:.2f}ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.1f}\u00b5s"
    return f"{ns:.0f}ns"


def pct_delta(base: float, test: float) -> str:
    if base == 0:
        return "n/a"
    d = (test - base) / base * 100
    c = "\033[32m" if d <= 0 else "\033[31m"
    return f"{c}{d:+.1f}%\033[0m"


def speedup_ratio(base: float, test: float) -> float:
    if test <= 0:
        return float("inf")
    return base / test


def speedup_tag(base: float, test: float) -> str:
    r = speedup_ratio(base, test)
    if r >= 1:
        return f"\033[32m{r:.2f}x faster\033[0m"
    return f"\033[31m{1 / r:.2f}x slower\033[0m"


def gate_badge(ratio: float) -> str:
    if ratio >= MIN_SPEEDUP:
        return f"\033[42;30m PASS {ratio:>7.1f}x \u2265 {MIN_SPEEDUP:.0f}x \033[0m"
    return f"\033[41;37m FAIL {ratio:>7.1f}x < {MIN_SPEEDUP:.0f}x \033[0m"


# ═══════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def print_header(
    suite_name: str,
    extra_lines: list[str] | None = None,
) -> None:
    """Print a standardised header block at the top of a benchmark suite."""
    print(f"\n\033[1m{W}\033[0m")
    print(f"\033[1m  {suite_name}\033[0m")
    print(f"\033[1m{W}\033[0m")
    print(f"  Python       : {sys.version.split()[0]}  "
          f"({'free-threaded' if FREE_THREADED else 'GIL-enabled'})")
    print(f"  Platform     : {platform.platform()}")
    print(f"  CPU (logical): {CPUS}   Physical: {PHYSICAL}   Workers: {WORKERS}")
    print(f"  Gate target  : \u2265 {MIN_SPEEDUP:.0f}x  (cthreading vs stdlib)")
    c_exts: list[str] = []
    for mod in ("_sync", "_threading", "_queue", "_monitoring", "_tasks"):
        try:
            __import__(f"cthreading.{mod}")
            c_exts.append(mod)
        except ImportError:
            pass
    print(f"  C extensions : {', '.join(c_exts) if c_exts else 'NONE'}")
    if extra_lines:
        for line in extra_lines:
            print(f"  {line}")
    print(f"\033[1m{W}\033[0m\n")


def section(title: str) -> None:
    print(f"\n\033[1;36m{W}\033[0m")
    print(f"\033[1;36m  {title}\033[0m")
    print(f"\033[1;36m{W}\033[0m")


def record(
    suite: str,
    name: str,
    std: Metrics,
    ct: Metrics,
    *,
    show_seq: bool = False,
) -> None:
    """Print a head-to-head comparison block AND append to ALL_RESULTS.

    Parameters
    ----------
    suite : str       – suite tag (e.g. "numpy", "simulated")
    name  : str       – human-readable benchmark name
    std   : Metrics   – stdlib / baseline result
    ct    : Metrics   – cthreading result
    show_seq : bool   – if True, display a 3-column layout with sequential baseline
    """
    pad = "    "
    ratio = speedup_ratio(std.elapsed, ct.elapsed)
    tag = speedup_tag(std.elapsed, ct.elapsed)
    badge = gate_badge(ratio)

    print(f"\n{pad}\033[1m{name}\033[0m  \u2014  {tag}  {badge}")

    if show_seq and ct.seq_elapsed > 0:
        _print_3col(pad, std, ct)
    else:
        _print_2col(pad, std, ct)

    ALL_RESULTS.append((suite, name, std, ct))


# ── internal column printers ──────────────────────────────────────

def _ok(val: bool) -> str:
    return "\033[32mPASS\033[0m" if val else "\033[31mFAIL\033[0m"


def _print_3col(pad: str, std: Metrics, ct: Metrics) -> None:
    """Sequential + stdlib + cthreading columns."""
    seq = ct.seq_elapsed
    hdr = (f"{pad}{'':28s}  {'sequential':>14s}  {'stdlib':>14s}  "
           f"{'cthreading':>14s}  {'ct vs std':>14s}")
    print(hdr)
    print(f"{pad}{THIN}")

    # elapsed
    print(f"{pad}{'elapsed':28s}  {seq:>12.4f}s  {std.elapsed:>12.4f}s  "
          f"{ct.elapsed:>12.4f}s  {pct_delta(std.elapsed, ct.elapsed):>22s}")

    # speedup vs sequential
    std_vs = speedup_ratio(seq, std.elapsed)
    ct_vs = speedup_ratio(seq, ct.elapsed)
    print(f"{pad}{'speedup vs sequential':28s}  {'\u2014':>14s}  "
          f"{std_vs:>13.2f}x  {ct_vs:>13.2f}x  "
          f"{pct_delta(std_vs, ct_vs):>22s}")

    # ops/sec (only if ops > 0)
    if std.ops > 0:
        seq_ops = std.ops / seq if seq > 0 else 0
        print(f"{pad}{'ops/sec':28s}  {fmt_num(seq_ops):>14s}  "
              f"{fmt_num(std.ops_per_sec):>14s}  {fmt_num(ct.ops_per_sec):>14s}  "
              f"{pct_delta(std.ops_per_sec, ct.ops_per_sec):>22s}")

    # parallel efficiency
    std_eff = std_vs / WORKERS * 100
    ct_eff = ct_vs / WORKERS * 100
    print(f"{pad}{'parallel efficiency':28s}  {'\u2014':>14s}  "
          f"{std_eff:>13.1f}%  {ct_eff:>13.1f}%")

    # correct
    print(f"{pad}{'correct':28s}  {'PASS':>14s}  "
          f"{_ok(std.correct):>22s}  {_ok(ct.correct):>22s}")

    # final value / checksums
    if std.final_value is not None:
        print(f"{pad}{'final value':28s}  {'\u2014':>14s}  "
              f"{str(std.final_value):>14s}  {str(ct.final_value):>14s}")

    # extra info
    all_keys = list(dict.fromkeys(list(std.extra.keys()) + list(ct.extra.keys())))
    for key in all_keys:
        sv = str(std.extra.get(key, "\u2014"))
        cv = str(ct.extra.get(key, "\u2014"))
        print(f"{pad}{key:28s}  {'\u2014':>14s}  {sv:>14s}  {cv:>14s}")


def _print_2col(pad: str, std: Metrics, ct: Metrics) -> None:
    """stdlib + cthreading columns (no sequential)."""
    hdr = (f"{pad}{'':28s}  {'stdlib':>14s}  {'cthreading':>14s}  "
           f"{'delta':>14s}")
    print(hdr)
    print(f"{pad}{THIN}")

    # elapsed
    print(f"{pad}{'elapsed':28s}  {std.elapsed:>12.6f}s  "
          f"{ct.elapsed:>12.6f}s  {pct_delta(std.elapsed, ct.elapsed):>22s}")

    # ops/sec + ns/op
    if std.ops > 0:
        print(f"{pad}{'ops/sec':28s}  {fmt_num(std.ops_per_sec):>14s}  "
              f"{fmt_num(ct.ops_per_sec):>14s}  "
              f"{pct_delta(std.ops_per_sec, ct.ops_per_sec):>22s}")
        print(f"{pad}{'ns/op':28s}  {fmt_ns(std.ns_per_op):>14s}  "
              f"{fmt_ns(ct.ns_per_op):>14s}  "
              f"{pct_delta(std.ns_per_op, ct.ns_per_op):>22s}")

    # latencies
    sl, cl = std.latency_stats(), ct.latency_stats()
    if sl and cl:
        for key, lbl in [("min_ns", "lat min"), ("median_ns", "lat p50"),
                         ("p95_ns", "lat p95"), ("p99_ns", "lat p99"),
                         ("max_ns", "lat max"), ("stdev_ns", "lat \u03c3")]:
            sv, cv = sl.get(key, 0), cl.get(key, 0)
            print(f"{pad}{lbl:28s}  {fmt_ns(sv):>14s}  "
                  f"{fmt_ns(cv):>14s}  {pct_delta(sv, cv):>22s}")

    # correct
    print(f"{pad}{'correct':28s}  {_ok(std.correct):>22s}  "
          f"{_ok(ct.correct):>22s}")

    # final value
    if std.final_value is not None:
        print(f"{pad}{'final value':28s}  {str(std.final_value):>14s}  "
              f"{str(ct.final_value):>14s}")

    # extra
    all_keys = list(dict.fromkeys(list(std.extra.keys()) + list(ct.extra.keys())))
    for key in all_keys:
        sv = str(std.extra.get(key, "\u2014"))
        cv = str(ct.extra.get(key, "\u2014"))
        print(f"{pad}{key:28s}  {sv:>14s}  {cv:>14s}")


# ═══════════════════════════════════════════════════════════════════
# SUITE + AGGREGATE SUMMARIES
# ═══════════════════════════════════════════════════════════════════

def print_suite_summary(suite: str) -> None:
    results = [(n, s, c) for su, n, s, c in ALL_RESULTS if su == suite]
    if not results:
        return
    _print_summary_table(
        f"SUITE SUMMARY \u2014 {suite}  ({len(results)} benchmarks, "
        f"gate \u2265 {MIN_SPEEDUP:.0f}x)",
        results,
    )


def print_final_aggregate() -> None:
    """Print the FINAL combined summary across ALL suites."""
    if not ALL_RESULTS:
        return

    # ordered unique suites
    suites: list[str] = []
    seen: set[str] = set()
    for su, *_ in ALL_RESULTS:
        if su not in seen:
            suites.append(su)
            seen.add(su)

    total = len(ALL_RESULTS)
    print(f"\n\033[1m{W}\033[0m")
    print(f"\033[1m  FINAL AGGREGATED RESULTS \u2014 ALL {total} BENCHMARKS  "
          f"(gate \u2265 {MIN_SPEEDUP:.0f}x)\033[0m")
    print(f"\033[1m{W}\033[0m")

    passed = failed = 0
    grand_std = grand_ct = grand_seq = 0.0

    hdr = (f"    {'Suite':<12s}  {'Benchmark':<34s}  {'seq':>8s}  "
           f"{'stdlib':>10s}  {'cthread':>10s}  {'vs seq':>7s}  "
           f"{'vs std':>7s}  {'gate':>6s}  {'ok':>3s}")
    print(f"\n{hdr}")
    sep = (f"    {'\u2500' * 12}  {'\u2500' * 34}  {'\u2500' * 8}  "
           f"{'\u2500' * 10}  {'\u2500' * 10}  {'\u2500' * 7}  "
           f"{'\u2500' * 7}  {'\u2500' * 6}  {'\u2500' * 3}")
    print(sep)

    for suite in suites:
        rows = [(n, s, c) for su, n, s, c in ALL_RESULTS if su == suite]
        sub_std = sub_ct = sub_seq = 0.0
        first = True
        for name, s, c in rows:
            suite_col = suite if first else ""
            first = False
            sub_std += s.elapsed
            sub_ct += c.elapsed
            sub_seq += c.seq_elapsed

            seq_s = f"{c.seq_elapsed:.2f}s" if c.seq_elapsed > 0 else "\u2014"
            r = speedup_ratio(s.elapsed, c.elapsed)
            vs_seq = (f"{speedup_ratio(c.seq_elapsed, c.elapsed):.1f}x"
                      if c.seq_elapsed > 0 else "\u2014")

            if r >= MIN_SPEEDUP:
                g = "\033[32mPASS\033[0m"
                passed += 1
            else:
                g = "\033[31mFAIL\033[0m"
                failed += 1

            ok = "\033[32m\u2713\033[0m" if s.correct and c.correct else "\033[31m\u2717\033[0m"
            print(f"    {suite_col:<12s}  {name:<34s}  {seq_s:>8s}  "
                  f"{s.elapsed:>8.4f}s  {c.elapsed:>8.4f}s  {vs_seq:>7s}  "
                  f"{r:>6.1f}x  {g:>14s}  {ok}")

        # suite subtotal
        grand_std += sub_std
        grand_ct += sub_ct
        grand_seq += sub_seq
        sub_r = speedup_ratio(sub_std, sub_ct)
        sub_vs = (f"{speedup_ratio(sub_seq, sub_ct):.1f}x"
                  if sub_seq > 0 else "\u2014")
        sub_seq_s = f"{sub_seq:.2f}s" if sub_seq > 0 else "\u2014"
        print(f"    {'':<12s}  {'\u2500\u2500 subtotal \u2500\u2500':<34s}  "
              f"{sub_seq_s:>8s}  {sub_std:>8.3f}s  {sub_ct:>8.3f}s  "
              f"{sub_vs:>7s}  {sub_r:>6.1f}x")

    # grand total
    print(sep)
    grand_r = speedup_ratio(grand_std, grand_ct)
    grand_vs = (f"{speedup_ratio(grand_seq, grand_ct):.1f}x"
                if grand_seq > 0 else "\u2014")
    grand_seq_s = f"{grand_seq:.2f}s" if grand_seq > 0 else "\u2014"
    print(f"    {'GRAND TOTAL':<12s}  {'':<34s}  {grand_seq_s:>8s}  "
          f"{grand_std:>8.3f}s  {grand_ct:>8.3f}s  {grand_vs:>7s}  "
          f"{grand_r:>6.1f}x")

    print(f"\n    \033[1mBenchmarks passed gate: {passed}/{passed + failed}"
          f"  (\u2265 {MIN_SPEEDUP:.0f}x)\033[0m")
    all_ok = all(s.correct and c.correct for _, _, s, c in ALL_RESULTS)
    print(f"    \033[1mAll correct: {'YES' if all_ok else 'NO'}\033[0m")
    print(f"\n\033[1m{W}\033[0m\n")


# ─── internal summary table ──────────────────────────────────────

def _print_summary_table(
    title: str,
    results: list[tuple[str, Metrics, Metrics]],
) -> None:
    section(title)

    passed = failed = 0
    hdr = (f"    {'Benchmark':<40s}  {'seq':>8s}  {'stdlib':>10s}  "
           f"{'cthread':>10s}  {'vs seq':>7s}  {'vs std':>7s}  "
           f"{'gate':>6s}  {'ok':>3s}")
    print(f"\n{hdr}")
    print(f"    {'\u2500' * 40}  {'\u2500' * 8}  {'\u2500' * 10}  "
          f"{'\u2500' * 10}  {'\u2500' * 7}  {'\u2500' * 7}  "
          f"{'\u2500' * 6}  {'\u2500' * 3}")

    for name, s, c in results:
        seq_s = f"{c.seq_elapsed:.2f}s" if c.seq_elapsed > 0 else "\u2014"
        r = speedup_ratio(s.elapsed, c.elapsed)
        vs_seq = (f"{speedup_ratio(c.seq_elapsed, c.elapsed):.1f}x"
                  if c.seq_elapsed > 0 and c.elapsed > 0 else "\u2014")

        if r >= MIN_SPEEDUP:
            g = "\033[42;30m PASS \033[0m"
            passed += 1
        else:
            g = "\033[41;37m FAIL \033[0m"
            failed += 1

        ok = "\033[32m\u2713\033[0m" if s.correct and c.correct else "\033[31m\u2717\033[0m"
        print(f"    {name:<40s}  {seq_s:>8s}  {s.elapsed:>8.4f}s  "
              f"{c.elapsed:>8.4f}s  {vs_seq:>7s}  {r:>6.1f}x  {g}  {ok}")

    # totals
    t_std = sum(s.elapsed for _, s, _ in results)
    t_ct = sum(c.elapsed for _, _, c in results)
    t_seq = sum(c.seq_elapsed for _, _, c in results if c.seq_elapsed > 0)
    t_r = speedup_ratio(t_std, t_ct)
    t_vs = f"{speedup_ratio(t_seq, t_ct):.1f}x" if t_seq > 0 else "\u2014"
    t_seq_s = f"{t_seq:.2f}s" if t_seq > 0 else "\u2014"

    print(f"    {'\u2500' * 40}  {'\u2500' * 8}  {'\u2500' * 10}  "
          f"{'\u2500' * 10}  {'\u2500' * 7}  {'\u2500' * 7}  "
          f"{'\u2500' * 6}  {'\u2500' * 3}")
    print(f"    {'TOTAL':<40s}  {t_seq_s:>8s}  {t_std:>8.3f}s  "
          f"{t_ct:>8.3f}s  {t_vs:>7s}  {t_r:>6.1f}x")

    print(f"\n    \033[1mPassed: {passed}   Failed: {failed}   "
          f"Total: {passed + failed}\033[0m")
    all_ok = all(s.correct and c.correct for _, s, c in results)
    print(f"    \033[1mAll correct: {'YES' if all_ok else 'NO'}\033[0m")


# ═══════════════════════════════════════════════════════════════════
# JSON EXPORT / IMPORT  (for run_all_benchmarks.py aggregation)
# ═══════════════════════════════════════════════════════════════════

def export_json(filepath: str) -> None:
    data = []
    for suite, name, std, ct in ALL_RESULTS:
        data.append({
            "suite": suite,
            "name": name,
            "stdlib": std.to_dict(),
            "cthreading": ct.to_dict(),
        })
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> list[tuple[str, str, Metrics, Metrics]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    results: list[tuple[str, str, Metrics, Metrics]] = []
    for entry in data:
        s = entry["stdlib"]
        c = entry["cthreading"]
        std = Metrics(
            label=s["label"], ops=s["ops"], elapsed=s["elapsed"],
            correct=s["correct"], final_value=s.get("final_value"),
            seq_elapsed=s.get("seq_elapsed", 0.0),
            extra=s.get("extra", {}),
        )
        ct = Metrics(
            label=c["label"], ops=c["ops"], elapsed=c["elapsed"],
            correct=c["correct"], final_value=c.get("final_value"),
            seq_elapsed=c.get("seq_elapsed", 0.0),
            extra=c.get("extra", {}),
        )
        results.append((entry["suite"], entry["name"], std, ct))
    return results
