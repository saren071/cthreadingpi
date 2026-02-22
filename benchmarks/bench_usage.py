from collections.abc import Callable
import time
import threading
import sys
from cthreading import Ghost

# Configuration
TARGET = 1_000_000
THREADS = 4
OPS_PER_THREAD = TARGET // THREADS

print("--- Overhead vs Stdlib ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Operations:     {TARGET:,}")
print(f"Threads:        {THREADS}")
print("-" * 75)
print(f"{'TEST NAME':<25} | {'TIME (s)':<10} | {'OPS/SEC':<12} | {'RESULT':<15}")
print("-" * 75)

def run_test(name: str, func: Callable[[dict[str, int], threading.Lock | Ghost | None], None], args_gen: Callable[[], tuple[int, dict[str, int]]], validator: Callable[[tuple[int, dict[str, int]]], tuple[bool, str]]) -> float:
    """
    Generic runner to handle thread spawning, timing, and validation.
    """
    threads: list[threading.Thread] = []
    args = args_gen() # Generate args specific to the test
    
    start_time = time.time_ns()
    
    for _ in range(THREADS):
        t = threading.Thread(target=func, args=args)
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    end_time = time.time_ns()
    duration_sec = (end_time - start_time) / 1e9
    ops_sec = TARGET / duration_sec
    
    valid, message = validator(args)
    status = "SAFE" if valid else message
    
    print(f"{name:<25} | {duration_sec:<10.4f} | {ops_sec:<12,.0f} | {status}")
    return duration_sec

# ------------------------------------------------------------------
# SCENARIO 1: The Race (Control Group)
# ------------------------------------------------------------------
def work_race(n: int, state: dict[str, int]) -> None:
    for _ in range(n):
        state['val'] += 1

def args_race():
    return (OPS_PER_THREAD, {'val': 0})

def validate_race(args: tuple[int, dict[str, int]]) -> tuple[bool, str]:
    val = args[1]['val']
    return val == TARGET, f"LOST {TARGET - val:,} OPS"

# ------------------------------------------------------------------
# SCENARIO 2: Standard Library Lock (The Baseline)
# ------------------------------------------------------------------
def work_std_lock(n: int, state: dict[str, int], lock: threading.Lock) -> None:
    for _ in range(n):
        with lock:
            state['val'] += 1

def args_std():
    return (OPS_PER_THREAD, {'val': 0}, threading.Lock())

def validate_std(args: tuple[int, dict[str, int]]) -> tuple[bool, str]:
    val = args[1]['val']
    return val == TARGET, f"CORRUPT {val}"

# ------------------------------------------------------------------
# SCENARIO 3: Ghost as a MUTEX (Context Manager)
# ------------------------------------------------------------------
# This tests purely how fast Ghost is at locking/unlocking.
# It still pays the penalty of Python dictionary lookups and Integer allocation.
def work_ghost_mutex(n: int, state: dict[str, int], lock: Ghost) -> None:
    for _ in range(n):
        with lock:
            state['val'] += 1

def args_ghost_mutex():
    return (OPS_PER_THREAD, {'val': 0}, Ghost())

def validate_ghost_mutex(args: tuple[int, dict[str, int]]) -> tuple[bool, str]:
    val = args[1]['val']
    return val == TARGET, f"CORRUPT {val}"

# ------------------------------------------------------------------
# SCENARIO 4: Ghost as an ATOMIC INTEGER (The Rocket)
# ------------------------------------------------------------------
# This utilizes the C-level `iadd` optimization.
# It bypasses the GIL for the math and bypasses Python int allocation.
def work_ghost_atomic(n: int, ghost: Ghost) -> None:
    for _ in range(n):
        ghost.add(1) # In C: atomic_fetch_add(ghost->acc, 1)

def args_ghost_atomic():
    # Initialize Ghost with 0 to enable Integer Mode
    return (OPS_PER_THREAD, Ghost(0)) 

def validate_ghost_atomic(args: tuple[int, Ghost]) -> tuple[bool, str]:
    val = args[1].get() # Use .get() to retrieve value
    return val == TARGET, f"CORRUPT {val}"


# --- EXECUTION ---

# 1. Baseline: How fast is unsafe code?
t_race = run_test("No Lock (Unsafe)", work_race, args_race, validate_race)

# 2. Standard Lock: The cost of safety in Python
t_std = run_test("threading.Lock (Std)", work_std_lock, args_std, validate_std)

# 3. Ghost Mutex: The cost of safety with your Mutex
t_ghost_mutex = run_test("Ghost (Used as Mutex)", work_ghost_mutex, args_ghost_mutex, validate_ghost_mutex)

# 4. Ghost Atomic: The cost of hardware atomics
t_ghost_atomic = run_test("Ghost (Atomic Integer)", work_ghost_atomic, args_ghost_atomic, validate_ghost_atomic)
