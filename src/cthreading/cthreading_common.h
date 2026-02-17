/*
 * src/cthreading/cthreading_common.h
 *
 * Shared types, macros, and forward declarations used across all
 * cthreading C extension modules.
 *
 * Performance-critical: uses native OS synchronization primitives
 * (SRWLOCK + CONDITION_VARIABLE on Windows, pthread on Unix) instead
 * of PyThread_type_lock which is a kernel semaphore on Windows.
 */

#ifndef CTHREADING_COMMON_H
#define CTHREADING_COMMON_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdatomic.h>
#include <stdint.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <unistd.h>
#  include <pthread.h>
#endif

/* ================================================================
 * NATIVE FAST-LOCK PRIMITIVES
 *
 * SRWLOCK on Windows is user-space (no kernel transition).
 * PyThread_type_lock is CreateSemaphore → kernel object → slow.
 * ================================================================ */

#ifdef _WIN32

typedef SRWLOCK ct_mutex_t;
#define ct_mutex_init(m)      InitializeSRWLock(m)
#define ct_mutex_destroy(m)   ((void)0)
#define ct_mutex_lock(m)      AcquireSRWLockExclusive(m)
#define ct_mutex_unlock(m)    ReleaseSRWLockExclusive(m)
#define ct_mutex_trylock(m)   TryAcquireSRWLockExclusive(m)

typedef CONDITION_VARIABLE ct_cond_t;
#define ct_cond_init(c)       InitializeConditionVariable(c)
#define ct_cond_destroy(c)    ((void)0)
#define ct_cond_signal(c)     WakeConditionVariable(c)
#define ct_cond_broadcast(c)  WakeAllConditionVariable(c)

static inline void
ct_cond_wait(ct_cond_t *c, ct_mutex_t *m)
{
    SleepConditionVariableSRW(c, m, INFINITE, 0);
}

static inline int
ct_cond_timedwait_ms(ct_cond_t *c, ct_mutex_t *m, DWORD ms)
{
    return SleepConditionVariableSRW(c, m, ms, 0) ? 1 : 0;
}

typedef CRITICAL_SECTION ct_rlock_t;
#define ct_rlock_init(r)      InitializeCriticalSection(r)
#define ct_rlock_destroy(r)   DeleteCriticalSection(r)
#define ct_rlock_lock(r)      EnterCriticalSection(r)
#define ct_rlock_unlock(r)    LeaveCriticalSection(r)
#define ct_rlock_trylock(r)   TryEnterCriticalSection(r)

#else /* POSIX */

typedef pthread_mutex_t ct_mutex_t;
#define ct_mutex_init(m)      pthread_mutex_init(m, NULL)
#define ct_mutex_destroy(m)   pthread_mutex_destroy(m)
#define ct_mutex_lock(m)      pthread_mutex_lock(m)
#define ct_mutex_unlock(m)    pthread_mutex_unlock(m)
#define ct_mutex_trylock(m)   (pthread_mutex_trylock(m) == 0)

typedef pthread_cond_t ct_cond_t;
#define ct_cond_init(c)       pthread_cond_init(c, NULL)
#define ct_cond_destroy(c)    pthread_cond_destroy(c)
#define ct_cond_signal(c)     pthread_cond_signal(c)
#define ct_cond_broadcast(c)  pthread_cond_broadcast(c)

static inline void
ct_cond_wait(ct_cond_t *c, ct_mutex_t *m)
{
    pthread_cond_wait(c, m);
}

static inline int
ct_cond_timedwait_ms(ct_cond_t *c, ct_mutex_t *m, unsigned long ms)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec  += ms / 1000;
    ts.tv_nsec += (ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec  += 1;
        ts.tv_nsec -= 1000000000L;
    }
    return pthread_cond_timedwait(c, m, &ts) == 0 ? 1 : 0;
}

typedef pthread_mutex_t ct_rlock_t;

static inline void ct_rlock_init(ct_rlock_t *r) {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(r, &attr);
    pthread_mutexattr_destroy(&attr);
}
#define ct_rlock_destroy(r)   pthread_mutex_destroy(r)
#define ct_rlock_lock(r)      pthread_mutex_lock(r)
#define ct_rlock_unlock(r)    pthread_mutex_unlock(r)
#define ct_rlock_trylock(r)   (pthread_mutex_trylock(r) == 0)

#endif /* _WIN32 */

/* ----------------------------------------------------------------
 * Monotonic clock — cross-platform millisecond timer
 * ---------------------------------------------------------------- */

static inline double
ct_time_ms(void)
{
#ifdef _WIN32
    return (double)GetTickCount64();
#else
    struct timespec _ts;
    clock_gettime(CLOCK_MONOTONIC, &_ts);
    return _ts.tv_sec * 1000.0 + _ts.tv_nsec / 1e6;
#endif
}

/* ----------------------------------------------------------------
 * Platform helpers
 * ---------------------------------------------------------------- */

static inline int
cthreading_cpu_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

/*
 * Return the number of **physical** CPU cores (ignoring hyper-threading).
 * For CPU-bound work, physical cores is the optimal worker count —
 * using logical cores causes oversubscription and cache thrashing.
 */
static inline int
cthreading_physical_cpu_count(void)
{
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformation(NULL, &len);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return cthreading_cpu_count();

    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
        (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(len);
    if (!buf)
        return cthreading_cpu_count();

    if (!GetLogicalProcessorInformation(buf, &len)) {
        free(buf);
        return cthreading_cpu_count();
    }

    int cores = 0;
    DWORD count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for (DWORD i = 0; i < count; i++) {
        if (buf[i].Relationship == RelationProcessorCore)
            cores++;
    }
    free(buf);
    return cores > 0 ? cores : cthreading_cpu_count();
#else
    /* Try /proc/cpuinfo for "cpu cores" on Linux */
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        int physical = 0;
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "cpu cores", 9) == 0) {
                char *colon = strchr(line, ':');
                if (colon) {
                    int n = atoi(colon + 1);
                    if (n > physical) physical = n;
                }
            }
        }
        fclose(f);
        if (physical > 0) return physical;
    }
    /* Fallback: assume 2 logical per physical */
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 1 ? (int)((n + 1) / 2) : 1;
#endif
}

/* ----------------------------------------------------------------
 * Cache-line isolation (false-sharing prevention)
 * ---------------------------------------------------------------- */

#define CT_CACHELINE 64

/* ----------------------------------------------------------------
 * Telemetry note
 * ---------------------------------------------------------------- */

/*
 * Each C extension module (.pyd / .so) is a separate shared library,
 * so they cannot share globals across DLL boundaries on Windows.
 * Every module that cares about telemetry defines its own local flag.
 */

/* ----------------------------------------------------------------
 * Task node — used by both threading.c and tasks.c
 * ---------------------------------------------------------------- */

struct FutureObject;

typedef struct TaskNode {
    PyObject       *callable;
    PyObject       *args;
    PyObject       *kwargs;
    struct FutureObject *future;   /* NULL = fire-and-forget */
    int             priority;
    int64_t         group_id;
    struct TaskNode *next;
} TaskNode;

/* ----------------------------------------------------------------
 * Thread pool state — NOW WITH NATIVE LOCKS
 * ---------------------------------------------------------------- */

/* Forward declare MapContext for pool-based parallel_map */
typedef struct MapContext MapContext;

typedef struct {
    /* --- Queue management (used together under lock) --- */
    ct_mutex_t          queue_lock;
    ct_cond_t           queue_cond;
    TaskNode           *queue_head;
    TaskNode           *queue_tail;
    Py_ssize_t          queue_size;
    TaskNode           *free_list;

    unsigned long      *worker_ids;
    Py_ssize_t          num_workers;
    Py_ssize_t          max_workers;
    atomic_int          shutdown;
    atomic_int          paused;

    /* Active map context (set during pool_map, NULL otherwise) */
    MapContext         *active_map;
    ct_cond_t           map_done_cond;

    /* --- Hot atomics on isolated cache lines --- */
    char                _pad0[CT_CACHELINE];
    atomic_int          sleeping_workers;
    char                _pad1[CT_CACHELINE - sizeof(atomic_int)];
    atomic_llong        tasks_submitted;
    char                _pad2[CT_CACHELINE - sizeof(atomic_llong)];
    atomic_llong        tasks_completed;
    char                _pad3[CT_CACHELINE - sizeof(atomic_llong)];
    atomic_llong        tasks_failed;
} PoolState;

/* ----------------------------------------------------------------
 * Batch state — defined in tasks.c
 * ---------------------------------------------------------------- */

typedef struct {
    PyObject_HEAD
    TaskNode   *head;
    TaskNode   *tail;
    Py_ssize_t  count;
    Py_ssize_t  flush_threshold;
    int         default_priority;
    int64_t     default_group;
    PyObject   *pool;
} BatchObject;

/* ----------------------------------------------------------------
 * Counter shard — lock-free value, lock only for telemetry
 * ---------------------------------------------------------------- */

typedef struct {
    atomic_ullong       access_count;
    atomic_ullong       contention_count;
    atomic_ullong       version;
    atomic_llong        value;
    char                _pad[64];   /* avoid false sharing between shards */
} CounterShard;

#endif /* CTHREADING_COMMON_H */
