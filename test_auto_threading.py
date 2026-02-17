#!/usr/bin/env python3
"""Test script to demonstrate auto-threading with cthreading."""

import time

from cthreading import (
    auto_threaded,
    parallel_map,
)


def worker(args: tuple[str, float]) -> None:
    """Worker function for ThreadPool."""
    name, delay = args
    print(f"Task {name} starting")
    time.sleep(delay)
    print(f"Task {name} finished")


def main() -> None:
    """Main function with auto-threaded patching."""
    print("Starting main function with auto-threaded patching.")

    # Sequential execution
    print("Running tasks sequentially...")
    start = time.time()
    for i in range(3):
        worker((f"Worker-{i}", 0.5))
    end = time.time()
    sequential_time = end - start
    print(f"Sequential time: {sequential_time:.2f}s")

    # Threaded execution
    print("Running tasks with parallel_map...")
    start = time.time()
    parallel_map(worker, [("Worker-0", 0.5), ("Worker-1", 0.5), ("Worker-2", 0.5)])
    end = time.time()
    threaded_time = end - start
    print(f"Threaded time: {threaded_time:.2f}s")

    speedup = sequential_time / threaded_time if threaded_time > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    print("All tasks completed. Main function complete.")


if __name__ == "__main__":
    auto_threaded(main)()
