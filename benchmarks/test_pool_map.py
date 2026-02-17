"""Quick smoke test for pool_map / pool_starmap / ThreadPool.map / ThreadPool.starmap"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cthreading import pool_map, pool_starmap, parallel_map, parallel_starmap, ThreadPool

# pool_map
r = pool_map(lambda x: x * x, range(10))
assert r == [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], f"pool_map failed: {r}"
print(f"pool_map OK: {r}")

# pool_starmap
r = pool_starmap(lambda a, b: a + b, [(1, 2), (3, 4), (5, 6)])
assert r == [3, 7, 11], f"pool_starmap failed: {r}"
print(f"pool_starmap OK: {r}")

# parallel_map (standalone, fresh threads)
r = parallel_map(lambda x: x * 2, range(5))
assert r == [0, 2, 4, 6, 8], f"parallel_map failed: {r}"
print(f"parallel_map OK: {r}")

# parallel_starmap (standalone, fresh threads)
r = parallel_starmap(lambda a, b: a * b, [(2, 3), (4, 5)])
assert r == [6, 20], f"parallel_starmap failed: {r}"
print(f"parallel_starmap OK: {r}")

# ThreadPool.map instance method
pool = ThreadPool(num_workers=4)
r = pool.map(lambda x: x + 10, range(8))
assert r == [10, 11, 12, 13, 14, 15, 16, 17], f"ThreadPool.map failed: {r}"
print(f"ThreadPool.map OK: {r}")

# ThreadPool.starmap instance method
r = pool.starmap(lambda a, b: a - b, [(10, 3), (20, 5), (30, 7)])
assert r == [7, 15, 23], f"ThreadPool.starmap failed: {r}"
print(f"ThreadPool.starmap OK: {r}")

pool.shutdown()
print("\nAll tests passed!")
