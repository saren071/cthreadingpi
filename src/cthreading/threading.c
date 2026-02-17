/*
 * src/cthreading/threading.c
 *
 * C-level thread pool with full lifecycle management.
 *   - ThreadPool   : manages N worker threads, task queue, submit/wrap
 *   - FutureObject : waitable result handle returned by submit/auto_thread
 *   - cpu_count()  : detect number of CPU cores
 *   - auto_thread  : decorator that submits calls to a global pool,
 *                    returns a Future for the result
 *
 * ADAPTIVE PARALLEL MAP:
 *   Workers activate via cascade wakeup — starts with 1 thread for tiny
 *   workloads, scales to pool->num_workers dynamically while processing.
 *   No Python-side tuning needed; all logic lives in C.
 *
 *   Cascade rule: after each chunk, if
 *     remaining_items  >  workers_active * ADAPTIVE_THRESHOLD
 *   AND sleeping workers exist AND we're below num_workers,
 *   signal one more sleeping worker to join.
 *
 * ADAPTIVE TASK QUEUE:
 *   submit() wakes workers proportional to queue depth.
 *   Idle workers sleep at zero CPU cost.
 *   As tasks pile up, more workers are woken automatically.
 *   As work drains, workers sleep again.
 *
 * FUTURE OBJECT:
 *   Every submit() and auto_thread() call returns a Future.
 *   future.result(timeout=None) blocks until the task completes,
 *   then returns the return value or re-raises the exception.
 *   future.done() / future.cancelled() are non-blocking polls.
 *   Fire-and-forget: just ignore the returned Future.
 *
 * Workers run entirely in C; Python callables are invoked with the GIL
 * re-acquired per task. The queue is a simple intrusive linked list
 * protected by a mutex + condition (signalled via a lock).
 *
 * NOTE: TaskNode in cthreading_common.h needs one new field:
 *
 *     struct FutureObject *future;   // NULL = fire-and-forget
 *
 * Add this to the TaskNode struct definition in cthreading_common.h.
 */

#include "cthreading_common.h"
#include <string.h>
#ifdef _WIN32
#  include <process.h>   /* _beginthreadex */
#endif

/*
 * ADAPTIVE_THRESHOLD: minimum items-per-active-worker required before
 * waking another worker.  Lower = more aggressive parallelism.
 * 4 means "wake another thread if each active worker has >4 items left".
 */
#define ADAPTIVE_THRESHOLD 4

/* ================================================================
 * FUTURE OBJECT
 *
 * Returned by submit() and auto_thread() calls.
 * Workers write the result/exception into the Future and signal
 * the condvar so any thread blocking in future.result() wakes up.
 *
 * States:
 *   FUTURE_PENDING    - not yet started or running
 *   FUTURE_FINISHED   - completed with a result
 *   FUTURE_EXCEPTION  - completed with an exception
 *   FUTURE_CANCELLED  - cancelled before execution
 * ================================================================ */

#define FUTURE_PENDING   0
#define FUTURE_FINISHED  1
#define FUTURE_EXCEPTION 2
#define FUTURE_CANCELLED 3

typedef struct FutureObject {
    PyObject_HEAD
    PyObject   *result;      /* return value of the callable              */
    PyObject   *exc_type;    /* exception type, or NULL                   */
    PyObject   *exc_value;   /* exception value, or NULL                  */
    PyObject   *exc_tb;      /* exception traceback, or NULL              */
    ct_mutex_t  mu;
    ct_cond_t   cv;
    atomic_int  state;       /* FUTURE_* constants above                  */
} FutureObject;

static PyTypeObject FutureType;

static PyObject *
Future_new_object(void)
{
    FutureObject *fut = PyObject_New(FutureObject, &FutureType);
    if (!fut) return NULL;
    ct_mutex_init(&fut->mu);
    ct_cond_init(&fut->cv);
    fut->result    = NULL;
    fut->exc_type  = NULL;
    fut->exc_value = NULL;
    fut->exc_tb    = NULL;
    atomic_init(&fut->state, FUTURE_PENDING);
    return (PyObject *)fut;
}

static void
Future_dealloc(FutureObject *self)
{
    Py_XDECREF(self->result);
    Py_XDECREF(self->exc_type);
    Py_XDECREF(self->exc_value);
    Py_XDECREF(self->exc_tb);
    ct_cond_destroy(&self->cv);
    ct_mutex_destroy(&self->mu);
    PyObject_Del(self);
}

/*
 * future.result(timeout=None)
 * Blocks until the future is done. Returns the result or re-raises
 * the stored exception. Returns None for fire-and-forget futures that
 * return None.
 */
static PyObject *
Future_result(FutureObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"timeout", NULL};
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &timeout))
        return NULL;

    int state = atomic_load_explicit(&self->state, memory_order_acquire);

    if (state == FUTURE_PENDING) {
        Py_BEGIN_ALLOW_THREADS
        ct_mutex_lock(&self->mu);
        if (timeout < 0.0) {
            while (atomic_load_explicit(&self->state, memory_order_acquire)
                   == FUTURE_PENDING)
            {
                ct_cond_wait(&self->cv, &self->mu);
            }
        } else {
            double deadline = ct_time_ms() + timeout * 1000.0;
            while (atomic_load_explicit(&self->state, memory_order_acquire)
                   == FUTURE_PENDING)
            {
                double remaining = deadline - ct_time_ms();
                if (remaining <= 0.0) break;
                unsigned long ms = (unsigned long)remaining;
                if (ms == 0) ms = 1;
                ct_cond_timedwait_ms(&self->cv, &self->mu, ms);
            }
        }
        ct_mutex_unlock(&self->mu);
        Py_END_ALLOW_THREADS
        state = atomic_load_explicit(&self->state, memory_order_acquire);
    }

    if (state == FUTURE_PENDING) {
        /* Timed out */
        PyErr_SetString(PyExc_TimeoutError, "future.result() timed out");
        return NULL;
    }
    if (state == FUTURE_CANCELLED) {
        PyErr_SetString(PyExc_RuntimeError, "future was cancelled");
        return NULL;
    }
    if (state == FUTURE_EXCEPTION) {
        /* Re-raise stored exception */
        PyObject *tp  = self->exc_type  ? self->exc_type  : Py_None;
        PyObject *val = self->exc_value ? self->exc_value : Py_None;
        PyObject *tb  = self->exc_tb    ? self->exc_tb    : Py_None;
        Py_INCREF(tp); Py_INCREF(val); Py_INCREF(tb);
        PyErr_Restore(tp, val, tb);
        return NULL;
    }
    /* FUTURE_FINISHED */
    PyObject *r = self->result ? self->result : Py_None;
    Py_INCREF(r);
    return r;
}

static PyObject *
Future_done(FutureObject *self, PyObject *Py_UNUSED(ignored))
{
    int state = atomic_load_explicit(&self->state, memory_order_acquire);
    if (state != FUTURE_PENDING) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Future_cancelled(FutureObject *self, PyObject *Py_UNUSED(ignored))
{
    if (atomic_load_explicit(&self->state, memory_order_acquire)
        == FUTURE_CANCELLED)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Future_cancel(FutureObject *self, PyObject *Py_UNUSED(ignored))
{
    int expected = FUTURE_PENDING;
    if (atomic_compare_exchange_strong_explicit(
            &self->state, &expected, FUTURE_CANCELLED,
            memory_order_acq_rel, memory_order_relaxed))
    {
        ct_mutex_lock(&self->mu);
        ct_cond_broadcast(&self->cv);
        ct_mutex_unlock(&self->mu);
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
Future_exception(FutureObject *self, PyObject *Py_UNUSED(ignored))
{
    int state = atomic_load_explicit(&self->state, memory_order_acquire);
    if (state == FUTURE_PENDING) {
        PyErr_SetString(PyExc_RuntimeError,
                        "exception() called before future completed");
        return NULL;
    }
    if (state == FUTURE_EXCEPTION && self->exc_value) {
        Py_INCREF(self->exc_value);
        return self->exc_value;
    }
    Py_RETURN_NONE;
}

/*
 * Internal: called by pool_worker to store result and wake waiters.
 * Steals result reference.
 */
static void
future_set_result(FutureObject *fut, PyObject *result)
{
    ct_mutex_lock(&fut->mu);
    fut->result = result;  /* steal ref */
    atomic_store_explicit(&fut->state, FUTURE_FINISHED, memory_order_release);
    ct_cond_broadcast(&fut->cv);
    ct_mutex_unlock(&fut->mu);
}

/*
 * Internal: store a Python exception captured with PyErr_Fetch.
 * Steals references to tp, val, tb.
 */
static void
future_set_exception(FutureObject *fut,
                     PyObject *tp, PyObject *val, PyObject *tb)
{
    ct_mutex_lock(&fut->mu);
    fut->exc_type  = tp;
    fut->exc_value = val;
    fut->exc_tb    = tb;
    atomic_store_explicit(&fut->state, FUTURE_EXCEPTION, memory_order_release);
    ct_cond_broadcast(&fut->cv);
    ct_mutex_unlock(&fut->mu);
}

static PyMethodDef Future_methods[] = {
    {"result",    (PyCFunction)Future_result,    METH_VARARGS | METH_KEYWORDS,
     "result(timeout=None) -> value  Block until done, return value or raise exception"},
    {"done",      (PyCFunction)Future_done,      METH_NOARGS,
     "done() -> bool  Return True if the future has completed"},
    {"cancelled", (PyCFunction)Future_cancelled, METH_NOARGS,
     "cancelled() -> bool  Return True if the future was cancelled"},
    {"cancel",    (PyCFunction)Future_cancel,    METH_NOARGS,
     "cancel() -> bool  Attempt to cancel; returns True if successful"},
    {"exception", (PyCFunction)Future_exception, METH_NOARGS,
     "exception() -> exc  Return stored exception or None"},
    {NULL}
};

static PyTypeObject FutureType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._threading.Future",
    .tp_doc       = "Waitable result handle returned by submit() and auto_thread().",
    .tp_basicsize = sizeof(FutureObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_dealloc   = (destructor)Future_dealloc,
    .tp_methods   = Future_methods,
};


/* ================================================================
 * WORKER THREAD FUNCTION
 * ================================================================ */

typedef struct {
    PoolState  *pool;
    Py_ssize_t  worker_index;
} WorkerArg;

/* ================================================================
 * MAP CONTEXT — used by pool-based parallel map
 * Workers steal work from this via atomic next_index.
 * Must be defined before pool_worker so it can access members.
 * ================================================================ */

struct MapContext {
    PyObject       *fn;
    PyObject       *items;       /* borrowed ref to a PyList            */
    PyObject      **results;     /* pre-allocated result slots          */
    Py_ssize_t      num_items;
    Py_ssize_t      num_workers; /* pool->num_workers cap               */
    int             starmap;
    /* --- Cache-line isolated hot atomics --- */
    char            _pad0[CT_CACHELINE];
    atomic_llong    next_index;
    char            _pad1[CT_CACHELINE - sizeof(atomic_llong)];
    atomic_llong    items_completed;
    char            _pad2[CT_CACHELINE - sizeof(atomic_llong)];
    atomic_int      has_error;
    char            _pad3[CT_CACHELINE - sizeof(atomic_int)];
    atomic_llong    workers_finished;
    char            _pad4[CT_CACHELINE - sizeof(atomic_llong)];
    atomic_int      workers_active;
};

/* Helper: process one map item (shared by pool_worker and standalone pmap) */
static inline void
map_process_item(PyObject *fn, PyObject *items, PyObject **results,
                 Py_ssize_t idx, int starmap, atomic_int *has_error)
{
    PyObject *item = PyList_GET_ITEM(items, idx);
    PyObject *result;
    if (starmap) {
        if (!PyTuple_Check(item)) {
            PyObject *tup = PySequence_Tuple(item);
            if (!tup) {
                PyErr_Clear();
                atomic_store_explicit(has_error, 1, memory_order_relaxed);
                Py_INCREF(Py_None);
                results[idx] = Py_None;
                return;
            }
            result = PyObject_Call(fn, tup, NULL);
            Py_DECREF(tup);
        } else {
            result = PyObject_Call(fn, item, NULL);
        }
    } else {
        result = PyObject_CallOneArg(fn, item);
    }

    if (result == NULL) {
        PyErr_Clear();
        atomic_store_explicit(has_error, 1, memory_order_relaxed);
        Py_INCREF(Py_None);
        results[idx] = Py_None;
    } else {
        results[idx] = result;  /* steals reference */
    }
}

static void
pool_worker(void *raw_arg)
{
    WorkerArg *warg = (WorkerArg *)raw_arg;
    PoolState *pool = warg->pool;
    Py_ssize_t worker_index = warg->worker_index;
    PyMem_RawFree(warg);

    (void)worker_index;

    PyGILState_STATE gstate = PyGILState_Ensure();

    for (;;) {
        TaskNode *task = NULL;
        MapContext *map = NULL;

        Py_BEGIN_ALLOW_THREADS
        ct_mutex_lock(&pool->queue_lock);
        while (pool->queue_head == NULL &&
               pool->active_map == NULL &&
               !atomic_load_explicit(&pool->shutdown, memory_order_acquire))
        {
            atomic_fetch_add_explicit(&pool->sleeping_workers, 1,
                                      memory_order_relaxed);
            ct_cond_wait(&pool->queue_cond, &pool->queue_lock);
            atomic_fetch_sub_explicit(&pool->sleeping_workers, 1,
                                      memory_order_relaxed);
        }

        /* Priority: map work > queue tasks */
        map = pool->active_map;
        if (!map) {
            task = pool->queue_head;
            if (task) {
                pool->queue_head = task->next;
                if (pool->queue_head == NULL)
                    pool->queue_tail = NULL;
                pool->queue_size--;
            }
        }
        ct_mutex_unlock(&pool->queue_lock);
        Py_END_ALLOW_THREADS

        /* ================================================================
         * ADAPTIVE POOL MAP — cascade work-stealing
         * ================================================================ */
        if (map) {
            atomic_fetch_add_explicit(&map->workers_active, 1,
                                      memory_order_relaxed);

            for (;;) {
                Py_ssize_t cur = (Py_ssize_t)atomic_load_explicit(
                    &map->next_index, memory_order_relaxed);
                Py_ssize_t remaining = map->num_items - cur;
                if (remaining <= 0)
                    break;

                int active = atomic_load_explicit(&map->workers_active,
                                                  memory_order_relaxed);
                if (active < 1) active = 1;
                Py_ssize_t chunk = remaining / (2 * active);
                if (chunk < 1)  chunk = 1;
                if (chunk > 64) chunk = 64;

                Py_ssize_t start = (Py_ssize_t)atomic_fetch_add_explicit(
                    &map->next_index, (long long)chunk, memory_order_relaxed);
                if (start >= map->num_items)
                    break;
                Py_ssize_t end = start + chunk;
                if (end > map->num_items) end = map->num_items;

                for (Py_ssize_t idx = start; idx < end; idx++) {
                    map_process_item(map->fn, map->items, map->results,
                                     idx, map->starmap, &map->has_error);
                }

                Py_ssize_t new_remaining = map->num_items -
                    (Py_ssize_t)atomic_load_explicit(
                        &map->next_index, memory_order_relaxed);
                int active_now = atomic_load_explicit(&map->workers_active,
                                                      memory_order_relaxed);
                int sleeping   = atomic_load_explicit(&pool->sleeping_workers,
                                                      memory_order_relaxed);

                if (new_remaining > (Py_ssize_t)(active_now * ADAPTIVE_THRESHOLD) &&
                    sleeping > 0 &&
                    active_now < (int)map->num_workers)
                {
                    Py_BEGIN_ALLOW_THREADS
                    ct_mutex_lock(&pool->queue_lock);
                    ct_cond_signal(&pool->queue_cond);
                    ct_mutex_unlock(&pool->queue_lock);
                    Py_END_ALLOW_THREADS
                }

                long long done = atomic_fetch_add_explicit(
                    &map->items_completed,
                    (long long)(end - start),
                    memory_order_release) + (long long)(end - start);
                if (done >= (long long)map->num_items) {
                    Py_BEGIN_ALLOW_THREADS
                    ct_mutex_lock(&pool->queue_lock);
                    ct_cond_signal(&pool->map_done_cond);
                    ct_mutex_unlock(&pool->queue_lock);
                    Py_END_ALLOW_THREADS
                }
            }

            atomic_fetch_sub_explicit(&map->workers_active, 1,
                                      memory_order_release);

            Py_BEGIN_ALLOW_THREADS
            ct_mutex_lock(&pool->queue_lock);
            while (pool->active_map == map &&
                   !atomic_load_explicit(&pool->shutdown, memory_order_acquire))
            {
                ct_cond_wait(&pool->queue_cond, &pool->queue_lock);
            }
            ct_mutex_unlock(&pool->queue_lock);
            Py_END_ALLOW_THREADS
            continue;
        }

        if (task == NULL) {
            /* Woken with no task and no map = shutdown */
            break;
        }

        /*
         * If this task was cancelled via its Future, skip execution
         * but still mark it so waiters unblock.
         */
        FutureObject *fut = (FutureObject *)task->future;
        if (fut) {
            int fstate = atomic_load_explicit(&fut->state, memory_order_acquire);
            if (fstate == FUTURE_CANCELLED) {
                /* Already cancelled — skip, don't overwrite state */
                goto task_cleanup;
            }
        }

        /* Execute the Python callable (GIL held) */
        PyObject *result = NULL;
        if (task->kwargs && task->kwargs != Py_None)
            result = PyObject_Call(task->callable, task->args, task->kwargs);
        else
            result = PyObject_CallObject(task->callable, task->args);

        if (fut) {
            if (result == NULL) {
                /* Capture exception info for re-raise in future.result() */
                PyObject *tp = NULL, *val = NULL, *tb = NULL;
                PyErr_Fetch(&tp, &val, &tb);
                PyErr_NormalizeException(&tp, &val, &tb);
                future_set_exception(fut, tp, val, tb);
                atomic_fetch_add_explicit(&pool->tasks_failed, 1,
                                          memory_order_relaxed);
            } else {
                future_set_result(fut, result); /* steals ref */
            }
        } else {
            /* No future — fire-and-forget */
            if (result == NULL) {
                PyErr_Clear();
                atomic_fetch_add_explicit(&pool->tasks_failed, 1,
                                          memory_order_relaxed);
            } else {
                Py_DECREF(result);
            }
        }

        atomic_fetch_add_explicit(&pool->tasks_completed, 1,
                                  memory_order_relaxed);

task_cleanup:
        Py_DECREF(task->callable);
        Py_DECREF(task->args);
        Py_XDECREF(task->kwargs);
        Py_XDECREF(task->future);   /* release Future ref held by task */
        task->future = NULL;

        /* Return task node to free-list for reuse */
        ct_mutex_lock(&pool->queue_lock);
        task->next = pool->free_list;
        pool->free_list = task;
        ct_mutex_unlock(&pool->queue_lock);
    }

    PyGILState_Release(gstate);
}

/* ================================================================
 * ADAPTIVE QUEUE WAKEUP
 *
 * Called after enqueuing a task. Decides how many sleeping workers
 * to wake based on the ratio of pending tasks to active workers.
 *
 *   pending = 1-2    → wake 1 worker (new work arrived, handle it)
 *   pending = 3-6    → wake 2 workers (small burst)
 *   pending = 7+     → wake up to pending/ADAPTIVE_THRESHOLD workers
 *
 * Never wakes more workers than are sleeping.
 * Never wakes more workers than the pool has.
 *
 * Called with queue_lock HELD.
 * ================================================================ */
static void
adaptive_queue_wakeup(PoolState *pool)
{
    int sleeping = atomic_load_explicit(&pool->sleeping_workers,
                                        memory_order_relaxed);
    if (sleeping <= 0)
        return;

    int pending = (int)pool->queue_size;
    if (pending <= 0)
        return;

    /* How many workers do we want active? */
    int desired = (pending + ADAPTIVE_THRESHOLD - 1) / ADAPTIVE_THRESHOLD;
    if (desired < 1) desired = 1;
    if (desired > (int)pool->num_workers) desired = (int)pool->num_workers;

    /* How many are already active? */
    int active = (int)pool->num_workers - sleeping;
    if (active < 0) active = 0;

    int to_wake = desired - active;
    if (to_wake <= 0) to_wake = 1;  /* always wake at least 1 for new work */
    if (to_wake > sleeping) to_wake = sleeping;

    for (int i = 0; i < to_wake; i++)
        ct_cond_signal(&pool->queue_cond);
}

/* ================================================================
 * THREADPOOL OBJECT
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    PoolState state;
} ThreadPoolObject;

static void
ThreadPool_dealloc(ThreadPoolObject *self)
{
    PoolState *pool = &self->state;

    ct_mutex_lock(&pool->queue_lock);
    atomic_store_explicit(&pool->shutdown, 1, memory_order_release);
    ct_cond_broadcast(&pool->queue_cond);
    ct_mutex_unlock(&pool->queue_lock);

    ct_mutex_lock(&pool->queue_lock);
    TaskNode *node = pool->queue_head;
    while (node) {
        TaskNode *next = node->next;
        Py_DECREF(node->callable);
        Py_DECREF(node->args);
        Py_XDECREF(node->kwargs);
        /* Cancel any pending futures so waiters unblock */
        if (node->future) {
            FutureObject *fut = (FutureObject *)node->future;
            int expected = FUTURE_PENDING;
            if (atomic_compare_exchange_strong_explicit(
                    &fut->state, &expected, FUTURE_CANCELLED,
                    memory_order_acq_rel, memory_order_relaxed))
            {
                ct_mutex_lock(&fut->mu);
                ct_cond_broadcast(&fut->cv);
                ct_mutex_unlock(&fut->mu);
            }
            Py_DECREF(node->future);
        }
        PyMem_RawFree(node);
        node = next;
    }
    pool->queue_head = pool->queue_tail = NULL;
    pool->queue_size = 0;

    TaskNode *fl = pool->free_list;
    while (fl) {
        TaskNode *next = fl->next;
        PyMem_RawFree(fl);
        fl = next;
    }
    pool->free_list = NULL;
    ct_mutex_unlock(&pool->queue_lock);

    ct_cond_destroy(&pool->map_done_cond);
    ct_cond_destroy(&pool->queue_cond);
    ct_mutex_destroy(&pool->queue_lock);
    if (pool->worker_ids)
        PyMem_Free(pool->worker_ids);

    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
ThreadPool_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"num_workers", NULL};
    Py_ssize_t num_workers = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &num_workers))
        return NULL;

    if (num_workers <= 0)
        num_workers = (Py_ssize_t)cthreading_cpu_count();

    ThreadPoolObject *self = (ThreadPoolObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    PoolState *pool = &self->state;
    memset(pool, 0, sizeof(PoolState));

    ct_mutex_init(&pool->queue_lock);
    ct_cond_init(&pool->queue_cond);
    ct_cond_init(&pool->map_done_cond);
    pool->active_map = NULL;

    pool->max_workers = num_workers;
    pool->worker_ids = (unsigned long *)PyMem_Calloc(
        (size_t)num_workers, sizeof(unsigned long));
    if (pool->worker_ids == NULL) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }

    atomic_init(&pool->shutdown, 0);
    atomic_init(&pool->paused, 0);
    atomic_init(&pool->sleeping_workers, 0);
    atomic_init(&pool->tasks_submitted, 0);
    atomic_init(&pool->tasks_completed, 0);
    atomic_init(&pool->tasks_failed, 0);

    for (Py_ssize_t i = 0; i < num_workers; i++) {
        WorkerArg *warg = (WorkerArg *)PyMem_RawCalloc(1, sizeof(WorkerArg));
        if (warg == NULL) {
            Py_DECREF(self);
            return PyErr_NoMemory();
        }
        warg->pool = pool;
        warg->worker_index = i;

        unsigned long tid = PyThread_start_new_thread(pool_worker, warg);
        if (tid == (unsigned long)-1) {
            PyMem_RawFree(warg);
            Py_DECREF(self);
            PyErr_SetString(PyExc_RuntimeError, "failed to start worker thread");
            return NULL;
        }
        pool->worker_ids[i] = tid;
        pool->num_workers = i + 1;
    }

    return (PyObject *)self;
}

/*
 * Internal submit — returns a new Future (with ref owned by caller),
 * or NULL on error. Does NOT increment tasks_submitted (caller does).
 */
static FutureObject *
_pool_submit_node(PoolState *pool,
                  PyObject *fn, PyObject *fn_args, PyObject *fn_kwargs)
{
    TaskNode *node = NULL;
    ct_mutex_lock(&pool->queue_lock);
    if (pool->free_list) {
        node = pool->free_list;
        pool->free_list = node->next;
    }
    ct_mutex_unlock(&pool->queue_lock);
    if (node == NULL) {
        node = (TaskNode *)PyMem_RawCalloc(1, sizeof(TaskNode));
        if (node == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
    }

    Py_INCREF(fn);
    node->callable = fn;

    if (fn_args == NULL || fn_args == Py_None) {
        node->args = PyTuple_New(0);
    } else if (PyTuple_Check(fn_args)) {
        Py_INCREF(fn_args);
        node->args = fn_args;
    } else {
        node->args = PySequence_Tuple(fn_args);
    }
    if (node->args == NULL) {
        Py_DECREF(fn);
        ct_mutex_lock(&pool->queue_lock);
        node->next = pool->free_list;
        pool->free_list = node;
        ct_mutex_unlock(&pool->queue_lock);
        return NULL;
    }

    if (fn_kwargs && fn_kwargs != Py_None) {
        Py_INCREF(fn_kwargs);
        node->kwargs = fn_kwargs;
    } else {
        node->kwargs = NULL;
    }

    /* Create future and attach to node */
    PyObject *fut_obj = Future_new_object();
    if (!fut_obj) {
        Py_DECREF(fn);
        Py_DECREF(node->args);
        Py_XDECREF(node->kwargs);
        ct_mutex_lock(&pool->queue_lock);
        node->next = pool->free_list;
        pool->free_list = node;
        ct_mutex_unlock(&pool->queue_lock);
        return NULL;
    }
    Py_INCREF(fut_obj);                              /* one ref owned by task node   */
    node->future = (struct FutureObject *)fut_obj;   /* task holds a ref             */
    node->next = NULL;
    node->priority = 0;
    node->group_id = 0;

    ct_mutex_lock(&pool->queue_lock);
    if (pool->queue_tail)
        pool->queue_tail->next = node;
    else
        pool->queue_head = node;
    pool->queue_tail = node;
    pool->queue_size++;
    adaptive_queue_wakeup(pool);  /* smart wakeup based on load   */
    ct_mutex_unlock(&pool->queue_lock);

    atomic_fetch_add_explicit(&pool->tasks_submitted, 1, memory_order_relaxed);

    /* Return the future — caller owns this reference */
    return (FutureObject *)fut_obj;
}

static PyObject *
ThreadPool_submit(ThreadPoolObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "args", "kwargs", "priority", "group", NULL};
    PyObject *fn;
    PyObject *fn_args   = NULL;
    PyObject *fn_kwargs = NULL;
    int       priority  = 0;
    long long group     = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOiL", kwlist,
                                     &fn, &fn_args, &fn_kwargs,
                                     &priority, &group))
        return NULL;

    if (!PyCallable_Check(fn)) {
        PyErr_SetString(PyExc_TypeError, "fn must be callable");
        return NULL;
    }

    PoolState *pool = &self->state;
    if (atomic_load_explicit(&pool->shutdown, memory_order_acquire)) {
        PyErr_SetString(PyExc_RuntimeError, "cannot submit to a shut-down pool");
        return NULL;
    }
    (void)priority; (void)group;

    FutureObject *fut = _pool_submit_node(pool, fn, fn_args, fn_kwargs);
    if (!fut) return NULL;
    return (PyObject *)fut;   /* caller owns ref */
}

static PyObject *
ThreadPool_wrap(ThreadPoolObject *self, PyObject *args)
{
    PyObject *fn;
    PyObject *fn_args = NULL;
    if (!PyArg_ParseTuple(args, "O|O", &fn, &fn_args))
        return NULL;
    if (!PyCallable_Check(fn)) {
        PyErr_SetString(PyExc_TypeError, "fn must be callable");
        return NULL;
    }
    PoolState *pool = &self->state;
    FutureObject *fut = _pool_submit_node(pool, fn, fn_args, NULL);
    if (!fut) return NULL;
    return (PyObject *)fut;
}

static PyObject *
ThreadPool_shutdown(ThreadPoolObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"wait", NULL};
    int wait = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p", kwlist, &wait))
        return NULL;

    PoolState *pool = &self->state;

    ct_mutex_lock(&pool->queue_lock);
    atomic_store_explicit(&pool->shutdown, 1, memory_order_release);
    ct_cond_broadcast(&pool->queue_cond);
    ct_mutex_unlock(&pool->queue_lock);

    if (wait) {
        Py_BEGIN_ALLOW_THREADS
        for (int tick = 0; tick < 500; tick++) {
            long long submitted = atomic_load_explicit(
                &pool->tasks_submitted, memory_order_acquire);
            long long completed = atomic_load_explicit(
                &pool->tasks_completed, memory_order_acquire);
            if (completed >= submitted)
                break;
#ifdef _WIN32
            Sleep(10);
#else
            usleep(10000);
#endif
        }
        Py_END_ALLOW_THREADS
    }

    Py_RETURN_NONE;
}

static PyObject *
ThreadPool_stats(ThreadPoolObject *self, PyObject *Py_UNUSED(ignored))
{
    PoolState *pool = &self->state;
    return Py_BuildValue("{s:n, s:n, s:L, s:L, s:L, s:i, s:i}",
        "num_workers",     pool->num_workers,
        "queue_size",      pool->queue_size,
        "tasks_submitted", atomic_load_explicit(&pool->tasks_submitted, memory_order_relaxed),
        "tasks_completed", atomic_load_explicit(&pool->tasks_completed, memory_order_relaxed),
        "tasks_failed",    atomic_load_explicit(&pool->tasks_failed, memory_order_relaxed),
        "sleeping_workers",atomic_load_explicit(&pool->sleeping_workers, memory_order_relaxed),
        "shutdown",        atomic_load_explicit(&pool->shutdown, memory_order_relaxed));
}

static PyObject *
ThreadPool_num_workers_get(ThreadPoolObject *self, void *Py_UNUSED(closure))
{
    return PyLong_FromSsize_t(self->state.num_workers);
}

static PyGetSetDef ThreadPool_getset[] = {
    {"num_workers", (getter)ThreadPool_num_workers_get, NULL,
     "Number of worker threads", NULL},
    {NULL}
};

/* Forward declarations */
static PyObject *_pool_map_impl(PoolState *pool, PyObject *fn,
                                PyObject *items_arg, int starmap);
static PyObject *ThreadPool_map(ThreadPoolObject *self,
                                PyObject *args, PyObject *kwds);
static PyObject *ThreadPool_starmap(ThreadPoolObject *self,
                                    PyObject *args, PyObject *kwds);

static PyMethodDef ThreadPool_methods[] = {
    {"submit",   (PyCFunction)ThreadPool_submit,   METH_VARARGS | METH_KEYWORDS,
     "submit(fn, args=None, kwargs=None) -> Future"},
    {"wrap",     (PyCFunction)ThreadPool_wrap,      METH_VARARGS,
     "wrap(fn, args=None) -> Future"},
    {"shutdown", (PyCFunction)ThreadPool_shutdown,  METH_VARARGS | METH_KEYWORDS,
     "Signal workers to stop"},
    {"stats",    (PyCFunction)ThreadPool_stats,     METH_NOARGS,
     "Get pool statistics"},
    {"map",      (PyCFunction)ThreadPool_map,       METH_VARARGS | METH_KEYWORDS,
     "Adaptive parallel map(fn, items)"},
    {"starmap",  (PyCFunction)ThreadPool_starmap,   METH_VARARGS | METH_KEYWORDS,
     "Adaptive parallel starmap(fn, items)"},
    {NULL}
};

static PyTypeObject ThreadPoolType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._threading.ThreadPool",
    .tp_doc       = "High-performance C-level thread pool with adaptive parallel map.",
    .tp_basicsize = sizeof(ThreadPoolObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new       = ThreadPool_new,
    .tp_dealloc   = (destructor)ThreadPool_dealloc,
    .tp_methods   = ThreadPool_methods,
    .tp_getset    = ThreadPool_getset,
};

/* ================================================================
 * MODULE-LEVEL FUNCTIONS
 * ================================================================ */

static PyObject *
mod_cpu_count(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    return PyLong_FromLong(cthreading_cpu_count());
}

static ThreadPoolObject *_default_pool = NULL;

static ThreadPoolObject *
get_default_pool(void)
{
    if (_default_pool == NULL) {
        PyObject *args = PyTuple_New(0);
        if (args == NULL) return NULL;
        _default_pool = (ThreadPoolObject *)ThreadPool_new(
            &ThreadPoolType, args, NULL);
        Py_DECREF(args);
    }
    return _default_pool;
}

static PyObject *
mod_set_default_pool_size(PyObject *Py_UNUSED(self), PyObject *args)
{
    Py_ssize_t n;
    if (!PyArg_ParseTuple(args, "n", &n))
        return NULL;
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "pool size must be > 0");
        return NULL;
    }

    if (_default_pool) {
        PyObject *empty = PyTuple_New(0);
        if (empty) {
            PyObject *r = ThreadPool_shutdown(_default_pool, empty, NULL);
            Py_XDECREF(r);
            Py_DECREF(empty);
        }
        Py_DECREF((PyObject *)_default_pool);
        _default_pool = NULL;
    }

    PyObject *pool_args = Py_BuildValue("(n)", n);
    if (pool_args == NULL)
        return NULL;
    _default_pool = (ThreadPoolObject *)ThreadPool_new(
        &ThreadPoolType, pool_args, NULL);
    Py_DECREF(pool_args);
    if (_default_pool == NULL)
        return NULL;

    Py_RETURN_NONE;
}

static PyObject *
mod_get_default_pool(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    ThreadPoolObject *pool = get_default_pool();
    if (pool == NULL)
        return NULL;
    Py_INCREF(pool);
    return (PyObject *)pool;
}

/* ================================================================
 * auto_thread DECORATOR
 *
 * Wraps a callable so every call is submitted to the default pool.
 * Returns a Future — caller can:
 *   result = my_func(args).result()   # block for return value
 *   my_func(args)                     # fire-and-forget (Future GC'd)
 *
 * Adaptive wakeup in _pool_submit_node() handles thread scaling:
 *   1 call queued  → 1 worker woken
 *   N calls queued → up to N/ADAPTIVE_THRESHOLD workers woken
 *   Workers sleep when queue drains → zero idle CPU cost
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    PyObject *wrapped;
} AutoThreadObject;

static int
AutoThread_traverse(AutoThreadObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->wrapped);
    return 0;
}

static int
AutoThread_clear(AutoThreadObject *self)
{
    Py_CLEAR(self->wrapped);
    return 0;
}

static void
AutoThread_dealloc(AutoThreadObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_XDECREF(self->wrapped);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
AutoThread_call(AutoThreadObject *self, PyObject *args, PyObject *kwds)
{
    ThreadPoolObject *pool = get_default_pool();
    if (pool == NULL)
        return NULL;

    PoolState *p = &pool->state;
    if (atomic_load_explicit(&p->shutdown, memory_order_acquire)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "auto_thread: default pool is shut down");
        return NULL;
    }

    FutureObject *fut = _pool_submit_node(
        p,
        self->wrapped,
        args   ? args : NULL,
        (kwds && kwds != Py_None && PyDict_Size(kwds) > 0) ? kwds : NULL);

    if (!fut) return NULL;
    return (PyObject *)fut;   /* caller gets a waitable Future */
}

static PyObject *
AutoThread_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", NULL};
    PyObject *fn;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &fn))
        return NULL;
    if (!PyCallable_Check(fn)) {
        PyErr_SetString(PyExc_TypeError, "auto_thread requires a callable");
        return NULL;
    }

    AutoThreadObject *self =
        (AutoThreadObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;
    Py_INCREF(fn);
    self->wrapped = fn;
    return (PyObject *)self;
}

/*
 * auto_thread.__wrapped__  — exposes the original function so
 * introspection tools (inspect, functools.wraps, etc.) work correctly.
 */
static PyObject *
AutoThread_get_wrapped(AutoThreadObject *self, void *Py_UNUSED(c))
{
    Py_INCREF(self->wrapped);
    return self->wrapped;
}

static PyGetSetDef AutoThread_getset[] = {
    {"__wrapped__", (getter)AutoThread_get_wrapped, NULL,
     "The original unwrapped function", NULL},
    {NULL}
};

static PyTypeObject AutoThreadType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._threading.auto_thread",
    .tp_doc       =
        "Decorator: calls are automatically submitted to the default pool.\n"
        "Returns a Future. Call .result() to block for the return value,\n"
        "or ignore it for fire-and-forget semantics.",
    .tp_basicsize = sizeof(AutoThreadObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_new       = AutoThread_new,
    .tp_dealloc   = (destructor)AutoThread_dealloc,
    .tp_traverse  = (traverseproc)AutoThread_traverse,
    .tp_clear     = (inquiry)AutoThread_clear,
    .tp_call      = (ternaryfunc)AutoThread_call,
    .tp_getset    = AutoThread_getset,
};

/* ================================================================
 * PARALLEL MAP (standalone threads, internal fallback)
 * ================================================================ */

typedef struct {
    PyObject       *fn;
    PyObject       *items;
    PyObject      **results;
    Py_ssize_t      num_items;
    int             starmap;
    char            _pad0[CT_CACHELINE];
    atomic_llong    next_index;
    char            _pad1[CT_CACHELINE - sizeof(atomic_llong)];
    atomic_int      has_error;
} PMapState;

static void
pmap_worker_adaptive(void *arg)
{
    PMapState *st = (PMapState *)arg;
    PyGILState_STATE gstate = PyGILState_Ensure();

    for (;;) {
        Py_ssize_t cur = (Py_ssize_t)atomic_load_explicit(
            &st->next_index, memory_order_relaxed);
        Py_ssize_t remaining = st->num_items - cur;
        Py_ssize_t chunk = remaining / 8;
        if (chunk < 1) chunk = 1;
        if (chunk > 64) chunk = 64;

        Py_ssize_t start = (Py_ssize_t)atomic_fetch_add_explicit(
            &st->next_index, (long long)chunk, memory_order_relaxed);
        if (start >= st->num_items)
            break;
        Py_ssize_t end = start + chunk;
        if (end > st->num_items) end = st->num_items;

        for (Py_ssize_t idx = start; idx < end; idx++) {
            map_process_item(st->fn, st->items, st->results,
                             idx, st->starmap, &st->has_error);
        }
    }

    PyGILState_Release(gstate);
}

#ifdef _WIN32
static unsigned __stdcall pmap_thread_entry(void *arg) {
    pmap_worker_adaptive(arg); return 0;
}
#else
static void *pmap_thread_entry(void *arg) {
    pmap_worker_adaptive(arg); return NULL;
}
#endif

static PyObject *
_parallel_map_impl(PyObject *fn, PyObject *items_arg,
                   Py_ssize_t num_workers, int starmap)
{
    if (!PyCallable_Check(fn)) {
        PyErr_SetString(PyExc_TypeError, "fn must be callable");
        return NULL;
    }

    PyObject *items = PySequence_List(items_arg);
    if (!items) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(items);
    if (n == 0) { Py_DECREF(items); return PyList_New(0); }

    if (num_workers <= 0)
        num_workers = (Py_ssize_t)cthreading_cpu_count();
    if (num_workers > n)
        num_workers = n;

    PyObject **results = (PyObject **)PyMem_RawCalloc(
        (size_t)n, sizeof(PyObject *));
    if (!results) { Py_DECREF(items); return PyErr_NoMemory(); }

    PMapState state;
    memset(&state, 0, sizeof(state));
    state.fn        = fn;
    state.items     = items;
    state.results   = results;
    state.num_items = n;
    state.starmap   = starmap;
    atomic_init(&state.next_index, 0);
    atomic_init(&state.has_error, 0);

    Py_INCREF(fn);

#ifdef _WIN32
    HANDLE *threads = (HANDLE *)PyMem_RawCalloc(
        (size_t)num_workers, sizeof(HANDLE));
    if (!threads) {
        Py_DECREF(fn); Py_DECREF(items); PyMem_RawFree(results);
        return PyErr_NoMemory();
    }
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < num_workers; i++)
        threads[i] = (HANDLE)_beginthreadex(
            NULL, 0, pmap_thread_entry, &state, 0, NULL);
    {
        Py_ssize_t remaining = num_workers, offset = 0;
        while (remaining > 0) {
            DWORD batch = (DWORD)(remaining > MAXIMUM_WAIT_OBJECTS
                                  ? MAXIMUM_WAIT_OBJECTS : remaining);
            WaitForMultipleObjects(batch, threads + offset, TRUE, INFINITE);
            offset += batch; remaining -= batch;
        }
    }
    for (Py_ssize_t i = 0; i < num_workers; i++) CloseHandle(threads[i]);
    Py_END_ALLOW_THREADS
    PyMem_RawFree(threads);
#else
    pthread_t *threads = (pthread_t *)PyMem_RawCalloc(
        (size_t)num_workers, sizeof(pthread_t));
    if (!threads) {
        Py_DECREF(fn); Py_DECREF(items); PyMem_RawFree(results);
        return PyErr_NoMemory();
    }
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < num_workers; i++)
        pthread_create(&threads[i], NULL, pmap_thread_entry, &state);
    for (Py_ssize_t i = 0; i < num_workers; i++)
        pthread_join(threads[i], NULL);
    Py_END_ALLOW_THREADS
    PyMem_RawFree(threads);
#endif

    Py_DECREF(fn);
    Py_DECREF(items);

    PyObject *result_list = PyList_New(n);
    if (!result_list) {
        for (Py_ssize_t i = 0; i < n; i++) Py_XDECREF(results[i]);
        PyMem_RawFree(results);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *r = results[i];
        if (r == NULL) { Py_INCREF(Py_None); r = Py_None; }
        PyList_SET_ITEM(result_list, i, r);
    }
    PyMem_RawFree(results);
    return result_list;
}

/* ================================================================
 * POOL-BASED ADAPTIVE MAP
 * ================================================================ */

static PyObject *
_build_result_list(PyObject **results, Py_ssize_t n)
{
    PyObject *result_list = PyList_New(n);
    if (!result_list) {
        for (Py_ssize_t i = 0; i < n; i++)
            Py_XDECREF(results[i]);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *r = results[i];
        if (r == NULL) { Py_INCREF(Py_None); r = Py_None; }
        PyList_SET_ITEM(result_list, i, r);
    }
    return result_list;
}

static PyObject *
_pool_map_impl(PoolState *pool, PyObject *fn, PyObject *items_arg, int starmap)
{
    if (!PyCallable_Check(fn)) {
        PyErr_SetString(PyExc_TypeError, "fn must be callable");
        return NULL;
    }

    if (atomic_load_explicit(&pool->shutdown, memory_order_acquire)) {
        PyErr_SetString(PyExc_RuntimeError, "pool is shut down");
        return NULL;
    }

    PyObject *items;
    if (PyList_CheckExact(items_arg)) {
        items = items_arg;
        Py_INCREF(items);
    } else {
        items = PySequence_List(items_arg);
    }
    if (!items) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(items);
    if (n == 0) { Py_DECREF(items); return PyList_New(0); }

    PyObject **results = (PyObject **)PyMem_RawCalloc(
        (size_t)n, sizeof(PyObject *));
    if (!results) { Py_DECREF(items); return PyErr_NoMemory(); }

    MapContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.fn          = fn;
    ctx.items       = items;
    ctx.results     = results;
    ctx.num_items   = n;
    ctx.num_workers = pool->num_workers;
    ctx.starmap     = starmap;
    atomic_init(&ctx.next_index, 0);
    atomic_init(&ctx.items_completed, 0);
    atomic_init(&ctx.has_error, 0);
    atomic_init(&ctx.workers_finished, 0);
    atomic_init(&ctx.workers_active, 0);

    Py_INCREF(fn);

    Py_BEGIN_ALLOW_THREADS

    ct_mutex_lock(&pool->queue_lock);
    pool->active_map = &ctx;

    int sleeping = atomic_load_explicit(&pool->sleeping_workers,
                                        memory_order_relaxed);
    if (sleeping > 0)
        ct_cond_signal(&pool->queue_cond);
    else
        ct_cond_broadcast(&pool->queue_cond);

    while (atomic_load_explicit(&ctx.items_completed, memory_order_acquire)
           < (long long)n &&
           !atomic_load_explicit(&pool->shutdown, memory_order_acquire))
    {
        ct_cond_wait(&pool->map_done_cond, &pool->queue_lock);
    }

    pool->active_map = NULL;
    ct_cond_broadcast(&pool->queue_cond);
    ct_mutex_unlock(&pool->queue_lock);

    Py_END_ALLOW_THREADS

    Py_DECREF(fn);
    Py_DECREF(items);

    PyObject *result_list = _build_result_list(results, n);
    PyMem_RawFree(results);
    return result_list;
}

static PyObject *
ThreadPool_map(ThreadPoolObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "items", NULL};
    PyObject *fn, *items_arg;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
                                     &fn, &items_arg))
        return NULL;
    return _pool_map_impl(&self->state, fn, items_arg, 0);
}

static PyObject *
ThreadPool_starmap(ThreadPoolObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "items", NULL};
    PyObject *fn, *items_arg;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
                                     &fn, &items_arg))
        return NULL;
    return _pool_map_impl(&self->state, fn, items_arg, 1);
}

static PyObject *
mod_parallel_map(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "items", "num_workers", NULL};
    PyObject *fn, *items_arg;
    Py_ssize_t num_workers = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|n", kwlist,
                                     &fn, &items_arg, &num_workers))
        return NULL;
    if (num_workers > 0 && _default_pool &&
        _default_pool->state.num_workers != num_workers) {
        PyObject *sz = Py_BuildValue("(n)", num_workers);
        if (sz) { PyObject *r = mod_set_default_pool_size(NULL, sz);
                  Py_XDECREF(r); Py_DECREF(sz); }
    }
    ThreadPoolObject *pool = get_default_pool();
    if (!pool) return NULL;
    return _pool_map_impl(&pool->state, fn, items_arg, 0);
}

static PyObject *
mod_auto_run_parallel(PyObject *self, PyObject *args, PyObject *kwds)
{
    return mod_parallel_map(self, args, kwds);
}

static PyObject *
mod_parallel_starmap(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "items", "num_workers", NULL};
    PyObject *fn, *items_arg;
    Py_ssize_t num_workers = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|n", kwlist,
                                     &fn, &items_arg, &num_workers))
        return NULL;
    if (num_workers > 0 && _default_pool &&
        _default_pool->state.num_workers != num_workers) {
        PyObject *sz = Py_BuildValue("(n)", num_workers);
        if (sz) { PyObject *r = mod_set_default_pool_size(NULL, sz);
                  Py_XDECREF(r); Py_DECREF(sz); }
    }
    ThreadPoolObject *pool = get_default_pool();
    if (!pool) return NULL;
    return _pool_map_impl(&pool->state, fn, items_arg, 1);
}

static PyObject *
mod_physical_cpu_count(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    return PyLong_FromLong(cthreading_physical_cpu_count());
}

static PyObject *
mod_pool_map(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "items", "num_workers", NULL};
    PyObject *fn, *items_arg;
    Py_ssize_t num_workers = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|n", kwlist,
                                     &fn, &items_arg, &num_workers))
        return NULL;
    if (num_workers > 0 && _default_pool &&
        _default_pool->state.num_workers != num_workers) {
        PyObject *sz = Py_BuildValue("(n)", num_workers);
        if (sz) { PyObject *r = mod_set_default_pool_size(NULL, sz);
                  Py_XDECREF(r); Py_DECREF(sz); }
    }
    ThreadPoolObject *pool = get_default_pool();
    if (!pool) return NULL;
    return _pool_map_impl(&pool->state, fn, items_arg, 0);
}

static PyObject *
mod_pool_starmap(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "items", "num_workers", NULL};
    PyObject *fn, *items_arg;
    Py_ssize_t num_workers = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|n", kwlist,
                                     &fn, &items_arg, &num_workers))
        return NULL;
    if (num_workers > 0 && _default_pool &&
        _default_pool->state.num_workers != num_workers) {
        PyObject *sz = Py_BuildValue("(n)", num_workers);
        if (sz) { PyObject *r = mod_set_default_pool_size(NULL, sz);
                  Py_XDECREF(r); Py_DECREF(sz); }
    }
    ThreadPoolObject *pool = get_default_pool();
    if (!pool) return NULL;
    return _pool_map_impl(&pool->state, fn, items_arg, 1);
}

/* ================================================================
 * THREAD REGISTRY
 * ================================================================ */

#define CT_MAX_THREADS 4096

typedef struct {
    ct_mutex_t  mu;
    PyObject   *threads[CT_MAX_THREADS];
    int         count;
    unsigned long main_tid;
    PyObject   *main_thread_obj;
} ThreadRegistry;

static ThreadRegistry _registry;

static void registry_init(void) {
    ct_mutex_init(&_registry.mu);
    _registry.count = 0;
    _registry.main_tid = PyThread_get_thread_ident();
    _registry.main_thread_obj = NULL;
}
static void registry_add(PyObject *t) {
    ct_mutex_lock(&_registry.mu);
    if (_registry.count < CT_MAX_THREADS) {
        Py_INCREF(t);
        _registry.threads[_registry.count++] = t;
    }
    ct_mutex_unlock(&_registry.mu);
}
static void registry_remove(PyObject *t) {
    ct_mutex_lock(&_registry.mu);
    for (int i = 0; i < _registry.count; i++) {
        if (_registry.threads[i] == t) {
            Py_DECREF(t);
            _registry.threads[i] = _registry.threads[--_registry.count];
            break;
        }
    }
    ct_mutex_unlock(&_registry.mu);
}

/* ================================================================
 * THREAD OBJECT
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    PyObject       *target;
    PyObject       *args;
    PyObject       *kwargs;
    PyObject       *name;
    ct_mutex_t      mu;
    ct_cond_t       cv;
    unsigned long   ident;
    unsigned long   native_id;
    int             started;
    int             alive;
    int             daemon;
    int             joined;
} ThreadObject;

static int Thread_traverse(ThreadObject *s, visitproc visit, void *arg) {
    Py_VISIT(s->target); Py_VISIT(s->args);
    Py_VISIT(s->kwargs); Py_VISIT(s->name); return 0;
}
static int Thread_clear_refs(ThreadObject *s) {
    Py_CLEAR(s->target); Py_CLEAR(s->args);
    Py_CLEAR(s->kwargs); Py_CLEAR(s->name); return 0;
}
static void Thread_dealloc(ThreadObject *s) {
    PyObject_GC_UnTrack(s);
    Thread_clear_refs(s);
    ct_cond_destroy(&s->cv);
    ct_mutex_destroy(&s->mu);
    Py_TYPE(s)->tp_free((PyObject *)s);
}

static PyTypeObject ThreadType;

static PyObject *
Thread_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"target","args","kwargs","name","daemon",NULL};
    PyObject *target = Py_None, *t_args = NULL, *t_kwargs = NULL;
    PyObject *name = NULL;
    int daemon = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOp", kwlist,
                                     &target, &t_args, &t_kwargs,
                                     &name, &daemon))
        return NULL;

    ThreadObject *self = (ThreadObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    ct_mutex_init(&self->mu);
    ct_cond_init(&self->cv);

    if (target != Py_None) { Py_INCREF(target); self->target = target; }
    else self->target = NULL;

    if (t_args) { Py_INCREF(t_args); self->args = t_args; }
    else { self->args = PyTuple_New(0); if (!self->args) { Py_DECREF(self); return NULL; } }

    if (t_kwargs && t_kwargs != Py_None) { Py_INCREF(t_kwargs); self->kwargs = t_kwargs; }
    else self->kwargs = NULL;

    if (name) { Py_INCREF(name); self->name = name; }
    else {
        static atomic_ullong tc = 0;
        unsigned long long n = atomic_fetch_add_explicit(&tc, 1, memory_order_relaxed);
        self->name = PyUnicode_FromFormat("Thread-%llu", n);
        if (!self->name) { Py_DECREF(self); return NULL; }
    }

    self->ident = 0; self->native_id = 0;
    self->started = 0; self->alive = 0;
    self->daemon = daemon; self->joined = 0;
    return (PyObject *)self;
}

static void
_thread_bootstrap(void *raw)
{
    ThreadObject *self = (ThreadObject *)raw;
    PyGILState_STATE gstate = PyGILState_Ensure();
#ifdef _WIN32
    self->native_id = (unsigned long)GetCurrentThreadId();
#else
    self->native_id = (unsigned long)pthread_self();
#endif
    PyObject *run = PyObject_GetAttrString((PyObject *)self, "run");
    if (run) { PyObject *r = PyObject_CallNoArgs(run); Py_XDECREF(r); Py_DECREF(run); }
    if (PyErr_Occurred()) PyErr_Print();
    ct_mutex_lock(&self->mu); self->alive = 0;
    ct_cond_broadcast(&self->cv); ct_mutex_unlock(&self->mu);
    registry_remove((PyObject *)self);
    Py_DECREF(self);
    PyGILState_Release(gstate);
}

static PyObject *
Thread_start(ThreadObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->started) {
        PyErr_SetString(PyExc_RuntimeError, "threads can only be started once");
        return NULL;
    }
    self->started = 1; self->alive = 1;
    Py_INCREF(self);
    registry_add((PyObject *)self);
    unsigned long ident = PyThread_start_new_thread(_thread_bootstrap, self);
    if (ident == (unsigned long)-1) {
        self->started = 0; self->alive = 0;
        registry_remove((PyObject *)self);
        Py_DECREF(self);
        PyErr_SetString(PyExc_RuntimeError, "can't start new thread");
        return NULL;
    }
    self->ident = ident;
    Py_RETURN_NONE;
}

static PyObject *
Thread_run(ThreadObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->target) {
        PyObject *r = self->kwargs
            ? PyObject_Call(self->target, self->args, self->kwargs)
            : PyObject_CallObject(self->target, self->args);
        Py_XDECREF(r);
        if (PyErr_Occurred()) return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *
Thread_join(ThreadObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"timeout", NULL};
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &timeout))
        return NULL;
    if (!self->started) {
        PyErr_SetString(PyExc_RuntimeError, "cannot join thread before it is started");
        return NULL;
    }
    if (self->ident == PyThread_get_thread_ident()) {
        PyErr_SetString(PyExc_RuntimeError, "cannot join current thread");
        return NULL;
    }
    ct_mutex_lock(&self->mu);
    if (timeout < 0) {
        while (self->alive) ct_cond_wait(&self->cv, &self->mu);
    } else {
        double deadline = ct_time_ms() + timeout * 1000.0;
        while (self->alive) {
            double remaining = deadline - ct_time_ms();
            if (remaining <= 0) break;
            unsigned long ms = (unsigned long)remaining;
            if (ms == 0) ms = 1;
            ct_cond_timedwait_ms(&self->cv, &self->mu, ms);
        }
    }
    self->joined = 1;
    ct_mutex_unlock(&self->mu);
    Py_RETURN_NONE;
}

static PyObject *Thread_is_alive(ThreadObject *s, PyObject *i) {
    (void)i; if (s->alive) Py_RETURN_TRUE; Py_RETURN_FALSE; }
static PyObject *Thread_get_name(ThreadObject *s, void *c) {
    (void)c; Py_INCREF(s->name); return s->name; }
static int Thread_set_name(ThreadObject *s, PyObject *v, void *c) {
    (void)c;
    if (!v || !PyUnicode_Check(v)) {
        PyErr_SetString(PyExc_TypeError, "name must be a string"); return -1; }
    Py_INCREF(v); Py_XDECREF(s->name); s->name = v; return 0; }
static PyObject *Thread_get_ident(ThreadObject *s, void *c) {
    (void)c; if (!s->started) Py_RETURN_NONE;
    return PyLong_FromUnsignedLong(s->ident); }
static PyObject *Thread_get_native_id(ThreadObject *s, void *c) {
    (void)c; if (!s->started) Py_RETURN_NONE;
    return PyLong_FromUnsignedLong(s->native_id); }
static PyObject *Thread_get_daemon(ThreadObject *s, void *c) {
    (void)c; if (s->daemon) Py_RETURN_TRUE; Py_RETURN_FALSE; }
static int Thread_set_daemon(ThreadObject *s, PyObject *v, void *c) {
    (void)c;
    if (s->started) {
        PyErr_SetString(PyExc_RuntimeError,
                        "cannot set daemon status of active thread"); return -1; }
    s->daemon = PyObject_IsTrue(v); return 0; }

static PyMethodDef Thread_methods[] = {
    {"start",    (PyCFunction)Thread_start,    METH_NOARGS,  "Start the thread"},
    {"run",      (PyCFunction)Thread_run,      METH_NOARGS,  "Thread activity"},
    {"join",     (PyCFunction)Thread_join,     METH_VARARGS | METH_KEYWORDS, "Wait"},
    {"is_alive", (PyCFunction)Thread_is_alive, METH_NOARGS,  "Alive?"},
    {NULL}
};
static PyGetSetDef Thread_getset[] = {
    {"name",      (getter)Thread_get_name,     (setter)Thread_set_name,   "Name",      NULL},
    {"ident",     (getter)Thread_get_ident,     NULL,                      "ID",        NULL},
    {"native_id", (getter)Thread_get_native_id, NULL,                      "Native ID", NULL},
    {"daemon",    (getter)Thread_get_daemon,    (setter)Thread_set_daemon, "Daemon",    NULL},
    {NULL}
};
static PyTypeObject ThreadType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._threading.Thread",
    .tp_basicsize = sizeof(ThreadObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = Thread_new,
    .tp_dealloc   = (destructor)Thread_dealloc,
    .tp_traverse  = (traverseproc)Thread_traverse,
    .tp_clear     = (inquiry)Thread_clear_refs,
    .tp_methods   = Thread_methods,
    .tp_getset    = Thread_getset,
};

/* ================================================================
 * TIMER OBJECT
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    PyObject       *function;
    PyObject       *args;
    PyObject       *kwargs;
    PyObject       *name;
    double          interval;
    ct_mutex_t      mu;
    ct_cond_t       cv;
    unsigned long   ident;
    unsigned long   native_id;
    int             started;
    int             alive;
    int             daemon;
    atomic_int      cancelled;
    int             finished;
} TimerObject;

static int Timer_traverse(TimerObject *s, visitproc visit, void *arg) {
    Py_VISIT(s->function); Py_VISIT(s->args);
    Py_VISIT(s->kwargs); Py_VISIT(s->name); return 0; }
static int Timer_clear_refs(TimerObject *s) {
    Py_CLEAR(s->function); Py_CLEAR(s->args);
    Py_CLEAR(s->kwargs); Py_CLEAR(s->name); return 0; }
static void Timer_dealloc(TimerObject *s) {
    PyObject_GC_UnTrack(s);
    Timer_clear_refs(s);
    ct_cond_destroy(&s->cv); ct_mutex_destroy(&s->mu);
    Py_TYPE(s)->tp_free((PyObject *)s); }

static PyTypeObject TimerType;

static PyObject *
Timer_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"interval","function","args","kwargs",NULL};
    double interval; PyObject *function;
    PyObject *t_args = NULL, *t_kwargs = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dO|OO", kwlist,
                                     &interval, &function, &t_args, &t_kwargs))
        return NULL;
    if (!PyCallable_Check(function)) {
        PyErr_SetString(PyExc_TypeError, "function must be callable");
        return NULL;
    }
    TimerObject *self = (TimerObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    ct_mutex_init(&self->mu); ct_cond_init(&self->cv);
    Py_INCREF(function); self->function = function;
    self->interval = interval;
    if (t_args) { Py_INCREF(t_args); self->args = t_args; }
    else { self->args = PyTuple_New(0); if (!self->args) { Py_DECREF(self); return NULL; } }
    if (t_kwargs && t_kwargs != Py_None) { Py_INCREF(t_kwargs); self->kwargs = t_kwargs; }
    else self->kwargs = NULL;
    static atomic_ullong tc = 0;
    unsigned long long n = atomic_fetch_add_explicit(&tc, 1, memory_order_relaxed);
    self->name = PyUnicode_FromFormat("Timer-%llu", n);
    if (!self->name) { Py_DECREF(self); return NULL; }
    self->ident = 0; self->native_id = 0;
    self->started = 0; self->alive = 0; self->daemon = 1;
    atomic_init(&self->cancelled, 0); self->finished = 0;
    return (PyObject *)self;
}

static void
_timer_bootstrap(void *raw)
{
    TimerObject *self = (TimerObject *)raw;
    PyGILState_STATE gstate = PyGILState_Ensure();
#ifdef _WIN32
    self->native_id = (unsigned long)GetCurrentThreadId();
#else
    self->native_id = (unsigned long)pthread_self();
#endif
    ct_mutex_lock(&self->mu);
    if (!atomic_load_explicit(&self->cancelled, memory_order_relaxed)) {
        unsigned long ms = (unsigned long)(self->interval * 1000.0);
        if (ms > 0) ct_cond_timedwait_ms(&self->cv, &self->mu, ms);
    }
    ct_mutex_unlock(&self->mu);
    if (!atomic_load_explicit(&self->cancelled, memory_order_relaxed)) {
        PyObject *r = self->kwargs
            ? PyObject_Call(self->function, self->args, self->kwargs)
            : PyObject_CallObject(self->function, self->args);
        Py_XDECREF(r);
        if (PyErr_Occurred()) PyErr_Print();
    }
    ct_mutex_lock(&self->mu);
    self->alive = 0; self->finished = 1;
    ct_cond_broadcast(&self->cv);
    ct_mutex_unlock(&self->mu);
    registry_remove((PyObject *)self);
    Py_DECREF(self);
    PyGILState_Release(gstate);
}

static PyObject *Timer_start(TimerObject *self, PyObject *Py_UNUSED(i)) {
    if (self->started) {
        PyErr_SetString(PyExc_RuntimeError, "threads can only be started once");
        return NULL; }
    self->started = 1; self->alive = 1;
    Py_INCREF(self); registry_add((PyObject *)self);
    unsigned long ident = PyThread_start_new_thread(_timer_bootstrap, self);
    if (ident == (unsigned long)-1) {
        self->started = 0; self->alive = 0;
        registry_remove((PyObject *)self); Py_DECREF(self);
        PyErr_SetString(PyExc_RuntimeError, "can't start new thread");
        return NULL; }
    self->ident = ident;
    Py_RETURN_NONE; }

static PyObject *Timer_cancel(TimerObject *self, PyObject *Py_UNUSED(i)) {
    atomic_store_explicit(&self->cancelled, 1, memory_order_relaxed);
    ct_mutex_lock(&self->mu); ct_cond_signal(&self->cv);
    ct_mutex_unlock(&self->mu); Py_RETURN_NONE; }

static PyObject *Timer_join(TimerObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"timeout", NULL};
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &timeout)) return NULL;
    if (!self->started) {
        PyErr_SetString(PyExc_RuntimeError, "cannot join thread before it is started");
        return NULL; }
    ct_mutex_lock(&self->mu);
    if (timeout < 0) {
        while (self->alive) ct_cond_wait(&self->cv, &self->mu);
    } else {
        double deadline = ct_time_ms() + timeout * 1000.0;
        while (self->alive) {
            double remaining = deadline - ct_time_ms();
            if (remaining <= 0) break;
            unsigned long ms = (unsigned long)remaining;
            if (ms == 0) ms = 1;
            ct_cond_timedwait_ms(&self->cv, &self->mu, ms);
        }
    }
    ct_mutex_unlock(&self->mu); Py_RETURN_NONE; }

static PyObject *Timer_is_alive(TimerObject *s, PyObject *i) {
    (void)i; if (s->alive) Py_RETURN_TRUE; Py_RETURN_FALSE; }
static PyObject *Timer_get_name(TimerObject *s, void *c) {
    (void)c; Py_INCREF(s->name); return s->name; }
static PyObject *Timer_get_daemon(TimerObject *s, void *c) {
    (void)c; if (s->daemon) Py_RETURN_TRUE; Py_RETURN_FALSE; }

static PyMethodDef Timer_methods[] = {
    {"start",    (PyCFunction)Timer_start,    METH_NOARGS,                  "Start"},
    {"cancel",   (PyCFunction)Timer_cancel,   METH_NOARGS,                  "Cancel"},
    {"join",     (PyCFunction)Timer_join,     METH_VARARGS | METH_KEYWORDS, "Wait"},
    {"is_alive", (PyCFunction)Timer_is_alive, METH_NOARGS,                  "Alive?"},
    {NULL}
};
static PyGetSetDef Timer_getset[] = {
    {"name",   (getter)Timer_get_name,   NULL, "Name",   NULL},
    {"daemon", (getter)Timer_get_daemon, NULL, "Daemon", NULL},
    {NULL}
};
static PyTypeObject TimerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._threading.Timer",
    .tp_basicsize = sizeof(TimerObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = Timer_new,
    .tp_dealloc   = (destructor)Timer_dealloc,
    .tp_traverse  = (traverseproc)Timer_traverse,
    .tp_clear     = (inquiry)Timer_clear_refs,
    .tp_methods   = Timer_methods,
    .tp_getset    = Timer_getset,
};

/* ================================================================
 * MODULE-LEVEL THREADING HELPERS
 * ================================================================ */

static PyObject *mod_active_count(PyObject *s, PyObject *a) {
    (void)s; (void)a;
    ct_mutex_lock(&_registry.mu);
    int count = _registry.count + 1;
    ct_mutex_unlock(&_registry.mu);
    return PyLong_FromLong(count); }

static PyObject *mod_current_thread(PyObject *s, PyObject *a) {
    (void)s; (void)a;
    unsigned long tid = PyThread_get_thread_ident();
    if (tid == _registry.main_tid && _registry.main_thread_obj) {
        Py_INCREF(_registry.main_thread_obj); return _registry.main_thread_obj; }
    ct_mutex_lock(&_registry.mu);
    for (int i = 0; i < _registry.count; i++) {
        PyObject *t = _registry.threads[i];
        if (Py_IS_TYPE(t, &ThreadType) && ((ThreadObject *)t)->ident == tid) {
            Py_INCREF(t); ct_mutex_unlock(&_registry.mu); return t; }
        if (Py_IS_TYPE(t, &TimerType) && ((TimerObject *)t)->ident == tid) {
            Py_INCREF(t); ct_mutex_unlock(&_registry.mu); return t; }
    }
    ct_mutex_unlock(&_registry.mu);
    if (_registry.main_thread_obj) {
        Py_INCREF(_registry.main_thread_obj); return _registry.main_thread_obj; }
    Py_RETURN_NONE; }

static PyObject *mod_main_thread(PyObject *s, PyObject *a) {
    (void)s; (void)a;
    if (_registry.main_thread_obj) {
        Py_INCREF(_registry.main_thread_obj); return _registry.main_thread_obj; }
    Py_RETURN_NONE; }

static PyObject *mod_enumerate_threads(PyObject *s, PyObject *a) {
    (void)s; (void)a;
    ct_mutex_lock(&_registry.mu);
    PyObject *list = PyList_New(_registry.count + 1);
    if (!list) { ct_mutex_unlock(&_registry.mu); return NULL; }
    if (_registry.main_thread_obj) {
        Py_INCREF(_registry.main_thread_obj);
        PyList_SET_ITEM(list, 0, _registry.main_thread_obj);
    } else {
        Py_INCREF(Py_None); PyList_SET_ITEM(list, 0, Py_None);
    }
    for (int i = 0; i < _registry.count; i++) {
        Py_INCREF(_registry.threads[i]);
        PyList_SET_ITEM(list, i + 1, _registry.threads[i]);
    }
    ct_mutex_unlock(&_registry.mu);
    return list; }

static PyObject *mod_get_ident(PyObject *s, PyObject *a) {
    (void)s; (void)a;
    return PyLong_FromUnsignedLong(PyThread_get_thread_ident()); }
static PyObject *mod_get_native_id(PyObject *s, PyObject *a) {
    (void)s; (void)a;
#ifdef _WIN32
    return PyLong_FromUnsignedLong((unsigned long)GetCurrentThreadId());
#else
    return PyLong_FromUnsignedLong((unsigned long)pthread_self());
#endif
}
static PyObject *mod_stack_size(PyObject *s, PyObject *args) {
    (void)s;
    Py_ssize_t new_size = 0;
    if (!PyArg_ParseTuple(args, "|n", &new_size)) return NULL;
    if (new_size > 0 && new_size < 32768) {
        PyErr_SetString(PyExc_ValueError, "size must be >= 32768"); return NULL; }
    return PyLong_FromSsize_t(0); }

#define CT_TIMEOUT_MAX (4294967.0)

/* ================================================================
 * MODULE DEFINITION
 * ================================================================ */

static void
threading_module_free(void *Py_UNUSED(mod))
{
    if (_default_pool) {
        PoolState *pool = &_default_pool->state;
        ct_mutex_lock(&pool->queue_lock);
        atomic_store_explicit(&pool->shutdown, 1, memory_order_release);
        ct_cond_broadcast(&pool->queue_cond);
        ct_mutex_unlock(&pool->queue_lock);
        Py_DECREF((PyObject *)_default_pool);
        _default_pool = NULL;
    }
}

static PyMethodDef threading_module_methods[] = {
    {"cpu_count",             (PyCFunction)mod_cpu_count,             METH_NOARGS,  "Logical CPU count"},
    {"physical_cpu_count",    (PyCFunction)mod_physical_cpu_count,    METH_NOARGS,  "Physical CPU count"},
    {"set_default_pool_size", (PyCFunction)mod_set_default_pool_size, METH_VARARGS, "Set default pool size"},
    {"get_default_pool",      (PyCFunction)mod_get_default_pool,      METH_NOARGS,  "Get default ThreadPool"},
    {"parallel_map",          (PyCFunction)mod_parallel_map,          METH_VARARGS | METH_KEYWORDS, "parallel_map(fn, items)"},
    {"auto_run_parallel",     (PyCFunction)mod_auto_run_parallel,     METH_VARARGS | METH_KEYWORDS, "alias of parallel_map"},
    {"parallel_starmap",      (PyCFunction)mod_parallel_starmap,      METH_VARARGS | METH_KEYWORDS, "parallel_starmap(fn, items)"},
    {"pool_map",              (PyCFunction)mod_pool_map,              METH_VARARGS | METH_KEYWORDS, "pool_map(fn, items)"},
    {"pool_starmap",          (PyCFunction)mod_pool_starmap,          METH_VARARGS | METH_KEYWORDS, "pool_starmap(fn, items)"},
    {"active_count",          (PyCFunction)mod_active_count,          METH_NOARGS,  "Alive thread count"},
    {"current_thread",        (PyCFunction)mod_current_thread,        METH_NOARGS,  "Current Thread"},
    {"main_thread",           (PyCFunction)mod_main_thread,           METH_NOARGS,  "Main Thread"},
    {"enumerate",             (PyCFunction)mod_enumerate_threads,     METH_NOARGS,  "All alive threads"},
    {"get_ident",             (PyCFunction)mod_get_ident,             METH_NOARGS,  "Thread ident"},
    {"get_native_id",         (PyCFunction)mod_get_native_id,         METH_NOARGS,  "Native thread ID"},
    {"stack_size",            (PyCFunction)mod_stack_size,            METH_VARARGS, "Stack size"},
    {NULL, NULL, 0, NULL},
};

static PyModuleDef threading_module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name    = "cthreading._threading",
    .m_doc     = "cthreading C core — ThreadPool, Future, auto_thread, adaptive parallel map",
    .m_size    = -1,
    .m_methods = threading_module_methods,
    .m_free    = (freefunc)threading_module_free,
};

PyMODINIT_FUNC
PyInit__threading(void)
{
    PyObject *m;

    if (PyType_Ready(&FutureType)     < 0) return NULL;
    if (PyType_Ready(&ThreadPoolType) < 0) return NULL;
    if (PyType_Ready(&AutoThreadType) < 0) return NULL;
    if (PyType_Ready(&ThreadType)     < 0) return NULL;
    if (PyType_Ready(&TimerType)      < 0) return NULL;

    m = PyModule_Create(&threading_module_def);
    if (m == NULL) return NULL;

    registry_init();

    /* Create MainThread sentinel */
    {
        PyObject *empty_args = PyTuple_New(0);
        PyObject *kw = PyDict_New();
        if (!empty_args || !kw) {
            Py_XDECREF(empty_args); Py_XDECREF(kw);
            Py_DECREF(m); return NULL;
        }
        PyObject *main_name = PyUnicode_FromString("MainThread");
        if (!main_name) {
            Py_DECREF(empty_args); Py_DECREF(kw);
            Py_DECREF(m); return NULL;
        }
        PyDict_SetItemString(kw, "name", main_name);
        Py_DECREF(main_name);
        _registry.main_thread_obj = Thread_new(&ThreadType, empty_args, kw);
        Py_DECREF(empty_args); Py_DECREF(kw);
        if (!_registry.main_thread_obj) { Py_DECREF(m); return NULL; }
        ThreadObject *mt = (ThreadObject *)_registry.main_thread_obj;
        mt->started = 1; mt->alive = 1;
        mt->ident = _registry.main_tid;
#ifdef _WIN32
        mt->native_id = (unsigned long)GetCurrentThreadId();
#else
        mt->native_id = (unsigned long)pthread_self();
#endif
    }

#define ADD_TYPE(name, typeobj) \
    Py_INCREF(&(typeobj)); \
    if (PyModule_AddObject(m, name, (PyObject *)&(typeobj)) < 0) { \
        Py_DECREF(&(typeobj)); Py_DECREF(m); return NULL; \
    }
    ADD_TYPE("Future",     FutureType);
    ADD_TYPE("ThreadPool", ThreadPoolType);
    ADD_TYPE("auto_thread",AutoThreadType);
    ADD_TYPE("Thread",     ThreadType);
    ADD_TYPE("Timer",      TimerType);
#undef ADD_TYPE

    if (PyModule_AddObject(m, "TIMEOUT_MAX",
                           PyFloat_FromDouble(CT_TIMEOUT_MAX)) < 0) {
        Py_DECREF(m); return NULL;
    }

#ifdef Py_MOD_GIL_NOT_USED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    return m;
}