/*
 * src/cthreading/threading.c
 *
 * C-level thread pool with full lifecycle management.
 *   - ThreadPool   : manages N worker threads, task queue, submit/wrap
 *   - cpu_count()  : detect number of CPU cores
 *   - auto_thread  : decorator that submits calls to a global pool
 *
 * Workers run entirely in C; Python callables are invoked with the GIL
 * re-acquired per task. The queue is a simple intrusive linked list
 * protected by a mutex + condition (signalled via a lock).
 */

#include "cthreading_common.h"
#include <string.h>
#ifdef _WIN32
#  include <process.h>   /* _beginthreadex */
#endif

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
    Py_ssize_t      num_workers; /* how many pool workers participate   */
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

    /*
     * Thread affinity: REMOVED.
     * Hard-pinning workers to single logical cores causes HT-sibling
     * cache contention on machines with hyper-threading (e.g. 20 logical /
     * 10 physical).  The OS scheduler does a better job of distributing
     * compute-heavy work across physical cores and NUMA nodes.
     */
    (void)worker_index;  /* suppress unused warning */

    PyGILState_STATE gstate = PyGILState_Ensure();

    for (;;) {
        TaskNode *task = NULL;
        MapContext *map = NULL;

        /* Release Python thread state while blocking on condvar */
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

        /* ---- Pool-based map: work-stealing with adaptive chunking ---- */
        if (map) {
            for (;;) {
                /* Adaptive chunk: large early, small late */
                Py_ssize_t cur = (Py_ssize_t)atomic_load_explicit(
                    &map->next_index, memory_order_relaxed);
                Py_ssize_t remaining = map->num_items - cur;
                Py_ssize_t chunk = remaining / (2 * map->num_workers);
                if (chunk < 1) chunk = 1;
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
                long long done = atomic_fetch_add_explicit(
                    &map->items_completed,
                    (long long)(end - start), memory_order_release) + (end - start);
                /* Signal caller when all items completed */
                if (done >= map->num_items) {
                    Py_BEGIN_ALLOW_THREADS
                    ct_mutex_lock(&pool->queue_lock);
                    ct_cond_signal(&pool->map_done_cond);
                    ct_mutex_unlock(&pool->queue_lock);
                    Py_END_ALLOW_THREADS
                }
            }

            /* Wait for caller to clear active_map before resuming */
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

        /* Execute the Python callable (thread state is held) */
        PyObject *result = NULL;
        if (task->kwargs && task->kwargs != Py_None)
            result = PyObject_Call(task->callable, task->args, task->kwargs);
        else
            result = PyObject_CallObject(task->callable, task->args);

        if (result == NULL) {
            PyErr_Clear();
            atomic_fetch_add_explicit(&pool->tasks_failed, 1,
                                      memory_order_relaxed);
        } else {
            Py_DECREF(result);
        }

        atomic_fetch_add_explicit(&pool->tasks_completed, 1,
                                  memory_order_relaxed);

        Py_DECREF(task->callable);
        Py_DECREF(task->args);
        Py_XDECREF(task->kwargs);

        /* Return task node to free-list for reuse */
        ct_mutex_lock(&pool->queue_lock);
        task->next = pool->free_list;
        pool->free_list = task;
        ct_mutex_unlock(&pool->queue_lock);
    }

    PyGILState_Release(gstate);
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

    /* Signal shutdown and wake all workers */
    ct_mutex_lock(&pool->queue_lock);
    atomic_store_explicit(&pool->shutdown, 1, memory_order_release);
    ct_cond_broadcast(&pool->queue_cond);
    ct_mutex_unlock(&pool->queue_lock);

    /* Drain remaining tasks */
    ct_mutex_lock(&pool->queue_lock);
    TaskNode *node = pool->queue_head;
    while (node) {
        TaskNode *next = node->next;
        Py_DECREF(node->callable);
        Py_DECREF(node->args);
        Py_XDECREF(node->kwargs);
        PyMem_RawFree(node);
        node = next;
    }
    pool->queue_head = pool->queue_tail = NULL;
    pool->queue_size = 0;

    /* Free the free-list */
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

    /* Spawn worker threads */
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

static PyObject *
ThreadPool_submit(ThreadPoolObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "args", "kwargs", "priority", "group", NULL};
    PyObject *fn;
    PyObject *fn_args = NULL;
    PyObject *fn_kwargs = NULL;
    int priority = 0;
    long long group = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOiL", kwlist,
                                     &fn, &fn_args, &fn_kwargs, &priority, &group))
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

    /* Try to reuse a node from the free-list */
    TaskNode *node = NULL;
    ct_mutex_lock(&pool->queue_lock);
    if (pool->free_list) {
        node = pool->free_list;
        pool->free_list = node->next;
    }
    ct_mutex_unlock(&pool->queue_lock);
    if (node == NULL) {
        node = (TaskNode *)PyMem_RawCalloc(1, sizeof(TaskNode));
        if (node == NULL)
            return PyErr_NoMemory();
    }

    Py_INCREF(fn);
    node->callable = fn;

    if (fn_args == NULL || fn_args == Py_None) {
        node->args = PyTuple_New(0);
    } else if (PyTuple_Check(fn_args)) {
        Py_INCREF(fn_args);
        node->args = fn_args;
    } else {
        /* Try to make it a tuple */
        node->args = PySequence_Tuple(fn_args);
        if (node->args == NULL) {
            Py_DECREF(fn);
            /* Return node to free-list */
            ct_mutex_lock(&pool->queue_lock);
            node->next = pool->free_list;
            pool->free_list = node;
            ct_mutex_unlock(&pool->queue_lock);
            return NULL;
        }
    }

    if (fn_kwargs && fn_kwargs != Py_None) {
        Py_INCREF(fn_kwargs);
        node->kwargs = fn_kwargs;
    } else {
        node->kwargs = NULL;
    }

    node->priority = priority;
    node->group_id = group;
    node->next = NULL;

    /* Enqueue and signal only if workers are sleeping */
    ct_mutex_lock(&pool->queue_lock);
    if (pool->queue_tail)
        pool->queue_tail->next = node;
    else
        pool->queue_head = node;
    pool->queue_tail = node;
    pool->queue_size++;
    int wake = atomic_load_explicit(&pool->sleeping_workers,
                                     memory_order_relaxed) > 0;
    ct_mutex_unlock(&pool->queue_lock);
    if (wake)
        ct_cond_signal(&pool->queue_cond);

    atomic_fetch_add_explicit(&pool->tasks_submitted, 1, memory_order_relaxed);

    Py_RETURN_NONE;
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

    PyObject *submit_args;
    if (fn_args && fn_args != Py_None) {
        submit_args = Py_BuildValue("(OO)", fn, fn_args);
    } else {
        submit_args = Py_BuildValue("(O)", fn);
    }
    if (submit_args == NULL)
        return NULL;

    PyObject *result = ThreadPool_submit(self, submit_args, NULL);
    Py_DECREF(submit_args);
    return result;
}

static PyObject *
ThreadPool_shutdown(ThreadPoolObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"wait", NULL};
    int wait = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p", kwlist, &wait))
        return NULL;

    PoolState *pool = &self->state;

    /* Set shutdown and wake ALL workers */
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
    return Py_BuildValue("{s:n, s:n, s:L, s:L, s:L, s:i}",
        "num_workers",     pool->num_workers,
        "queue_size",      pool->queue_size,
        "tasks_submitted", atomic_load_explicit(&pool->tasks_submitted, memory_order_relaxed),
        "tasks_completed", atomic_load_explicit(&pool->tasks_completed, memory_order_relaxed),
        "tasks_failed",    atomic_load_explicit(&pool->tasks_failed, memory_order_relaxed),
        "shutdown",        atomic_load_explicit(&pool->shutdown, memory_order_relaxed));
}

static PyObject *
ThreadPool_num_workers_get(ThreadPoolObject *self, void *Py_UNUSED(closure))
{
    return PyLong_FromSsize_t(self->state.num_workers);
}

static PyGetSetDef ThreadPool_getset[] = {
    {"num_workers", (getter)ThreadPool_num_workers_get, NULL, "Number of worker threads", NULL},
    {NULL}
};

/* Forward declarations for pool-based map (defined after parallel_map section) */
static PyObject *_pool_map_impl(PoolState *pool, PyObject *fn, PyObject *items_arg, int starmap);
static PyObject *ThreadPool_map(ThreadPoolObject *self, PyObject *args, PyObject *kwds);
static PyObject *ThreadPool_starmap(ThreadPoolObject *self, PyObject *args, PyObject *kwds);

static PyMethodDef ThreadPool_methods[] = {
    {"submit",   (PyCFunction)ThreadPool_submit,   METH_VARARGS | METH_KEYWORDS, "Submit a callable to the pool"},
    {"wrap",     (PyCFunction)ThreadPool_wrap,      METH_VARARGS,                 "Submit fn(args) to the pool (fire-and-forget)"},
    {"shutdown", (PyCFunction)ThreadPool_shutdown,  METH_VARARGS | METH_KEYWORDS, "Signal workers to stop"},
    {"stats",    (PyCFunction)ThreadPool_stats,     METH_NOARGS,                  "Get pool statistics"},
    {"map",      (PyCFunction)ThreadPool_map,       METH_VARARGS | METH_KEYWORDS, "Pool-based parallel map(fn, items)"},
    {"starmap",  (PyCFunction)ThreadPool_starmap,   METH_VARARGS | METH_KEYWORDS, "Pool-based parallel starmap(fn, items)"},
    {NULL}
};

static PyTypeObject ThreadPoolType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._threading.ThreadPool",
    .tp_doc       = "High-performance C-level thread pool with task queue.",
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

/* Global default pool (lazy) */
static ThreadPoolObject *_default_pool = NULL;

static ThreadPoolObject *
get_default_pool(void)
{
    if (_default_pool == NULL) {
        PyObject *args = PyTuple_New(0);
        if (args == NULL) return NULL;
        _default_pool = (ThreadPoolObject *)ThreadPool_new(&ThreadPoolType, args, NULL);
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

    /* Shutdown existing pool if any */
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
    _default_pool = (ThreadPoolObject *)ThreadPool_new(&ThreadPoolType, pool_args, NULL);
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

/* auto_thread: a decorator that wraps calls to submit to the default pool */
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

    /* Delegate to ThreadPool_submit — single code path for task enqueue */
    PyObject *submit_args;
    if (kwds && kwds != Py_None && PyDict_Size(kwds) > 0) {
        submit_args = Py_BuildValue("(OOO)", self->wrapped,
                                    args ? args : PyTuple_New(0), kwds);
    } else if (args) {
        submit_args = Py_BuildValue("(OO)", self->wrapped, args);
    } else {
        submit_args = Py_BuildValue("(O)", self->wrapped);
    }
    if (submit_args == NULL)
        return NULL;

    PyObject *result = ThreadPool_submit(pool, submit_args, NULL);
    Py_DECREF(submit_args);
    return result;
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

    AutoThreadObject *self = (AutoThreadObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;
    Py_INCREF(fn);
    self->wrapped = fn;
    return (PyObject *)self;
}

static PyTypeObject AutoThreadType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._threading.auto_thread",
    .tp_doc       = "Decorator: calls are automatically submitted to the default pool.",
    .tp_basicsize = sizeof(AutoThreadObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .tp_new       = AutoThread_new,
    .tp_dealloc   = (destructor)AutoThread_dealloc,
    .tp_traverse  = (traverseproc)AutoThread_traverse,
    .tp_clear     = (inquiry)AutoThread_clear,
    .tp_call      = (ternaryfunc)AutoThread_call,
};

/* ================================================================
 * PARALLEL MAP — work-stealing parallel execution
 * ================================================================ */

typedef struct {
    PyObject       *fn;
    PyObject       *items;       /* borrowed ref to a PyList            */
    PyObject      **results;     /* pre-allocated result slots          */
    Py_ssize_t      num_items;
    int             starmap;     /* if true, unpack tuple args          */
    /* --- Cache-line isolated hot atomics --- */
    char            _pad0[CT_CACHELINE];
    atomic_llong    next_index;  /* work-stealing counter               */
    char            _pad1[CT_CACHELINE - sizeof(atomic_llong)];
    atomic_int      has_error;
} PMapState;

/* Standalone pmap_worker with adaptive chunking (used by _parallel_map_impl) */
static void
pmap_worker_adaptive(void *arg)
{
    PMapState *st = (PMapState *)arg;
    PyGILState_STATE gstate = PyGILState_Ensure();

    for (;;) {
        /* Adaptive chunk: large early, small late */
        Py_ssize_t cur = (Py_ssize_t)atomic_load_explicit(
            &st->next_index, memory_order_relaxed);
        Py_ssize_t remaining = st->num_items - cur;
        Py_ssize_t chunk = remaining / 8;  /* ~12.5% of remaining */
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

/* Platform wrappers for standalone pmap threads */
#ifdef _WIN32
static unsigned __stdcall pmap_thread_entry(void *arg) {
    pmap_worker_adaptive(arg);
    return 0;
}
#else
static void *pmap_thread_entry(void *arg) {
    pmap_worker_adaptive(arg);
    return NULL;
}
#endif

/*
 * Core implementation shared by parallel_map and parallel_starmap.
 * Spawns fresh threads (no pool). For pool-based version, see _pool_map_impl.
 *   fn        : callable
 *   items_arg : iterable of items
 *   num_workers: 0 = physical_cpu_count()
 *   starmap   : 1 = call fn(*item), 0 = call fn(item)
 */
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
    if (n == 0) {
        Py_DECREF(items);
        return PyList_New(0);
    }

    if (num_workers <= 0)
        num_workers = (Py_ssize_t)cthreading_cpu_count();
    if (num_workers > n)
        num_workers = n;

    PyObject **results = (PyObject **)PyMem_RawCalloc(
        (size_t)n, sizeof(PyObject *));
    if (!results) {
        Py_DECREF(items);
        return PyErr_NoMemory();
    }

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
        Py_DECREF(fn);
        Py_DECREF(items);
        PyMem_RawFree(results);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < num_workers; i++) {
        threads[i] = (HANDLE)_beginthreadex(
            NULL, 0, pmap_thread_entry, &state, 0, NULL);
    }
    /* Wait in batches of MAXIMUM_WAIT_OBJECTS (64) */
    {
        Py_ssize_t remaining = num_workers;
        Py_ssize_t offset = 0;
        while (remaining > 0) {
            DWORD batch = (DWORD)(remaining > MAXIMUM_WAIT_OBJECTS
                                  ? MAXIMUM_WAIT_OBJECTS : remaining);
            WaitForMultipleObjects(batch, threads + offset, TRUE, INFINITE);
            offset    += batch;
            remaining -= batch;
        }
    }
    for (Py_ssize_t i = 0; i < num_workers; i++)
        CloseHandle(threads[i]);
    Py_END_ALLOW_THREADS

    PyMem_RawFree(threads);

#else  /* POSIX */
    pthread_t *threads = (pthread_t *)PyMem_RawCalloc(
        (size_t)num_workers, sizeof(pthread_t));
    if (!threads) {
        Py_DECREF(fn);
        Py_DECREF(items);
        PyMem_RawFree(results);
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

    /* Build result list */
    PyObject *result_list = PyList_New(n);
    if (!result_list) {
        for (Py_ssize_t i = 0; i < n; i++)
            Py_XDECREF(results[i]);
        PyMem_RawFree(results);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *r = results[i];
        if (r == NULL) {
            Py_INCREF(Py_None);
            r = Py_None;
        }
        PyList_SET_ITEM(result_list, i, r);  /* steals reference */
    }
    PyMem_RawFree(results);

    return result_list;
}

/*
 * parallel_map / parallel_starmap now route through the persistent
 * default pool for zero thread-creation overhead.  The standalone
 * _parallel_map_impl (fresh threads) is kept only as an internal
 * fallback and is not exposed to Python.
 */
static PyObject *
mod_parallel_map(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "items", "num_workers", NULL};
    PyObject *fn, *items_arg;
    Py_ssize_t num_workers = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|n", kwlist,
                                     &fn, &items_arg, &num_workers))
        return NULL;

    /* Resize default pool if num_workers specified */
    if (num_workers > 0 && _default_pool &&
        _default_pool->state.num_workers != num_workers) {
        PyObject *size_args = Py_BuildValue("(n)", num_workers);
        if (size_args) {
            PyObject *r = mod_set_default_pool_size(NULL, size_args);
            Py_XDECREF(r);
            Py_DECREF(size_args);
        }
    }

    ThreadPoolObject *pool = get_default_pool();
    if (pool == NULL) return NULL;
    return _pool_map_impl(&pool->state, fn, items_arg, 0);
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
        PyObject *size_args = Py_BuildValue("(n)", num_workers);
        if (size_args) {
            PyObject *r = mod_set_default_pool_size(NULL, size_args);
            Py_XDECREF(r);
            Py_DECREF(size_args);
        }
    }

    ThreadPoolObject *pool = get_default_pool();
    if (pool == NULL) return NULL;
    return _pool_map_impl(&pool->state, fn, items_arg, 1);
}

static PyObject *
mod_physical_cpu_count(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    return PyLong_FromLong(cthreading_physical_cpu_count());
}

/* ================================================================
 * POOL-BASED PARALLEL MAP — zero thread-creation overhead
 * Uses the persistent pool workers with work-stealing + adaptive chunking.
 * ================================================================ */

/* Helper: collect results array into a Python list */
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
        if (r == NULL) {
            Py_INCREF(Py_None);
            r = Py_None;
        }
        PyList_SET_ITEM(result_list, i, r);
    }
    return result_list;
}

/*
 * Pool-based parallel map: dispatches work to existing pool workers.
 * No thread creation/join overhead. Workers wake, steal work, then sleep.
 */
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

    /* Fast path: borrow if already a list, otherwise copy once */
    PyObject *items;
    if (PyList_CheckExact(items_arg)) {
        items = items_arg;
        Py_INCREF(items);
    } else {
        items = PySequence_List(items_arg);
    }
    if (!items) return NULL;

    Py_ssize_t n = PyList_GET_SIZE(items);
    if (n == 0) {
        Py_DECREF(items);
        return PyList_New(0);
    }

    PyObject **results = (PyObject **)PyMem_RawCalloc(
        (size_t)n, sizeof(PyObject *));
    if (!results) {
        Py_DECREF(items);
        return PyErr_NoMemory();
    }

    /* Build MapContext on the stack (it lives until workers finish) */
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

    Py_INCREF(fn);

    /* Install map context, wake all workers, wait for completion */
    Py_BEGIN_ALLOW_THREADS

    ct_mutex_lock(&pool->queue_lock);
    pool->active_map = &ctx;
    ct_cond_broadcast(&pool->queue_cond);  /* wake all sleeping workers */

    /* Wait until all items have been processed */
    while (atomic_load_explicit(&ctx.items_completed, memory_order_acquire)
           < (long long)n &&
           !atomic_load_explicit(&pool->shutdown, memory_order_acquire))
    {
        ct_cond_wait(&pool->map_done_cond, &pool->queue_lock);
    }

    /* Clear map and release workers back to normal loop */
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

/* ThreadPool.map(fn, items) — instance method */
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

/* ThreadPool.starmap(fn, items) — instance method */
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

/* Module-level pool_map/pool_starmap — use default pool */
static PyObject *
mod_pool_map(PyObject *Py_UNUSED(self), PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "items", "num_workers", NULL};
    PyObject *fn, *items_arg;
    Py_ssize_t num_workers = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|n", kwlist,
                                     &fn, &items_arg, &num_workers))
        return NULL;

    /* Resize default pool if num_workers specified */
    if (num_workers > 0 && _default_pool &&
        _default_pool->state.num_workers != num_workers) {
        PyObject *size_args = Py_BuildValue("(n)", num_workers);
        if (size_args) {
            PyObject *r = mod_set_default_pool_size(NULL, size_args);
            Py_XDECREF(r);
            Py_DECREF(size_args);
        }
    }

    ThreadPoolObject *pool = get_default_pool();
    if (pool == NULL) return NULL;
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
        PyObject *size_args = Py_BuildValue("(n)", num_workers);
        if (size_args) {
            PyObject *r = mod_set_default_pool_size(NULL, size_args);
            Py_XDECREF(r);
            Py_DECREF(size_args);
        }
    }

    ThreadPoolObject *pool = get_default_pool();
    if (pool == NULL) return NULL;
    return _pool_map_impl(&pool->state, fn, items_arg, 1);
}

/* ================================================================
 * THREAD REGISTRY — tracks all live Thread objects
 * ================================================================ */

#define CT_MAX_THREADS 4096

typedef struct {
    ct_mutex_t  mu;
    PyObject   *threads[CT_MAX_THREADS];   /* weak-ish refs: borrowed */
    int         count;
    unsigned long main_tid;
    PyObject   *main_thread_obj;           /* strong ref to sentinel */
} ThreadRegistry;

static ThreadRegistry _registry;

static void
registry_init(void)
{
    ct_mutex_init(&_registry.mu);
    _registry.count = 0;
    _registry.main_tid = PyThread_get_thread_ident();
    _registry.main_thread_obj = NULL;
}

static void
registry_add(PyObject *thread)
{
    ct_mutex_lock(&_registry.mu);
    if (_registry.count < CT_MAX_THREADS) {
        Py_INCREF(thread);
        _registry.threads[_registry.count++] = thread;
    }
    ct_mutex_unlock(&_registry.mu);
}

static void
registry_remove(PyObject *thread)
{
    ct_mutex_lock(&_registry.mu);
    for (int i = 0; i < _registry.count; i++) {
        if (_registry.threads[i] == thread) {
            Py_DECREF(thread);
            _registry.threads[i] = _registry.threads[--_registry.count];
            break;
        }
    }
    ct_mutex_unlock(&_registry.mu);
}

/* ================================================================
 * THREAD — standalone thread with start/join/is_alive
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

static int
Thread_traverse(ThreadObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->target);
    Py_VISIT(self->args);
    Py_VISIT(self->kwargs);
    Py_VISIT(self->name);
    return 0;
}

static int
Thread_clear_refs(ThreadObject *self)
{
    Py_CLEAR(self->target);
    Py_CLEAR(self->args);
    Py_CLEAR(self->kwargs);
    Py_CLEAR(self->name);
    return 0;
}

static void
Thread_dealloc(ThreadObject *self)
{
    PyObject_GC_UnTrack(self);
    Thread_clear_refs(self);
    ct_cond_destroy(&self->cv);
    ct_mutex_destroy(&self->mu);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* Forward declaration */
static PyTypeObject ThreadType;

static PyObject *
Thread_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"target", "args", "kwargs", "name", "daemon", NULL};
    PyObject *target = Py_None;
    PyObject *t_args = NULL;
    PyObject *t_kwargs = NULL;
    PyObject *name = NULL;
    int daemon = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OOOOp", kwlist,
                                     &target, &t_args, &t_kwargs, &name, &daemon))
        return NULL;

    ThreadObject *self = (ThreadObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    ct_mutex_init(&self->mu);
    ct_cond_init(&self->cv);

    if (target != Py_None) {
        Py_INCREF(target);
        self->target = target;
    } else {
        self->target = NULL;
    }

    if (t_args) {
        Py_INCREF(t_args);
        self->args = t_args;
    } else {
        self->args = PyTuple_New(0);
        if (!self->args) { Py_DECREF(self); return NULL; }
    }

    if (t_kwargs && t_kwargs != Py_None) {
        Py_INCREF(t_kwargs);
        self->kwargs = t_kwargs;
    } else {
        self->kwargs = NULL;
    }

    if (name) {
        Py_INCREF(name);
        self->name = name;
    } else {
        static atomic_ullong thread_counter = 0;
        unsigned long long n = atomic_fetch_add_explicit(&thread_counter, 1, memory_order_relaxed);
        self->name = PyUnicode_FromFormat("Thread-%llu", n);
        if (!self->name) { Py_DECREF(self); return NULL; }
    }

    self->ident = 0;
    self->native_id = 0;
    self->started = 0;
    self->alive = 0;
    self->daemon = daemon;
    self->joined = 0;

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

    /* Call self.run() — allows subclass override */
    PyObject *run = PyObject_GetAttrString((PyObject *)self, "run");
    if (run) {
        PyObject *result = PyObject_CallNoArgs(run);
        Py_XDECREF(result);
        Py_DECREF(run);
    }
    if (PyErr_Occurred()) {
        /* Print and clear — mirrors threading.excepthook default */
        PyErr_Print();
    }

    /* Mark dead, signal join waiters */
    ct_mutex_lock(&self->mu);
    self->alive = 0;
    ct_cond_broadcast(&self->cv);
    ct_mutex_unlock(&self->mu);

    registry_remove((PyObject *)self);
    Py_DECREF(self);   /* release the ref held by the running thread */

    PyGILState_Release(gstate);
}

static PyObject *
Thread_start(ThreadObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->started) {
        PyErr_SetString(PyExc_RuntimeError, "threads can only be started once");
        return NULL;
    }
    self->started = 1;
    self->alive = 1;   /* set BEFORE spawning so join() always blocks */

    Py_INCREF(self);   /* ref for the new thread */
    registry_add((PyObject *)self);

    unsigned long ident = PyThread_start_new_thread(_thread_bootstrap, (void *)self);
    if (ident == (unsigned long)-1) {
        self->started = 0;
        self->alive = 0;
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
        PyObject *result;
        if (self->kwargs)
            result = PyObject_Call(self->target, self->args, self->kwargs);
        else
            result = PyObject_CallObject(self->target, self->args);
        Py_XDECREF(result);
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
        while (self->alive)
            ct_cond_wait(&self->cv, &self->mu);
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

static PyObject *
Thread_is_alive(ThreadObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->alive)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Thread_get_name(ThreadObject *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->name);
    return self->name;
}

static int
Thread_set_name(ThreadObject *self, PyObject *value, void *Py_UNUSED(closure))
{
    if (!value || !PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "name must be a string");
        return -1;
    }
    Py_INCREF(value);
    Py_XDECREF(self->name);
    self->name = value;
    return 0;
}

static PyObject *
Thread_get_ident(ThreadObject *self, void *Py_UNUSED(closure))
{
    if (!self->started)
        Py_RETURN_NONE;
    return PyLong_FromUnsignedLong(self->ident);
}

static PyObject *
Thread_get_native_id(ThreadObject *self, void *Py_UNUSED(closure))
{
    if (!self->started)
        Py_RETURN_NONE;
    return PyLong_FromUnsignedLong(self->native_id);
}

static PyObject *
Thread_get_daemon(ThreadObject *self, void *Py_UNUSED(closure))
{
    if (self->daemon) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static int
Thread_set_daemon(ThreadObject *self, PyObject *value, void *Py_UNUSED(closure))
{
    if (self->started) {
        PyErr_SetString(PyExc_RuntimeError, "cannot set daemon status of active thread");
        return -1;
    }
    self->daemon = PyObject_IsTrue(value);
    return 0;
}

static PyMethodDef Thread_methods[] = {
    {"start",    (PyCFunction)Thread_start,    METH_NOARGS,                  "Start the thread"},
    {"run",      (PyCFunction)Thread_run,      METH_NOARGS,                  "Thread activity (override in subclass)"},
    {"join",     (PyCFunction)Thread_join,     METH_VARARGS | METH_KEYWORDS, "Wait for thread to finish"},
    {"is_alive", (PyCFunction)Thread_is_alive, METH_NOARGS,                  "Return whether the thread is alive"},
    {NULL}
};

static PyGetSetDef Thread_getset[] = {
    {"name",      (getter)Thread_get_name,      (setter)Thread_set_name,      "Thread name", NULL},
    {"ident",     (getter)Thread_get_ident,      NULL,                        "Thread identifier", NULL},
    {"native_id", (getter)Thread_get_native_id,  NULL,                        "Native thread ID", NULL},
    {"daemon",    (getter)Thread_get_daemon,     (setter)Thread_set_daemon,   "Daemon flag", NULL},
    {NULL}
};

static PyTypeObject ThreadType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._threading.Thread",
    .tp_doc       = "C-backed thread — start/join/is_alive with native OS threads.",
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
 * TIMER — delayed-execution thread
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

static int
Timer_traverse(TimerObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->function);
    Py_VISIT(self->args);
    Py_VISIT(self->kwargs);
    Py_VISIT(self->name);
    return 0;
}

static int
Timer_clear_refs(TimerObject *self)
{
    Py_CLEAR(self->function);
    Py_CLEAR(self->args);
    Py_CLEAR(self->kwargs);
    Py_CLEAR(self->name);
    return 0;
}

static void
Timer_dealloc(TimerObject *self)
{
    PyObject_GC_UnTrack(self);
    Timer_clear_refs(self);
    ct_cond_destroy(&self->cv);
    ct_mutex_destroy(&self->mu);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyTypeObject TimerType;  /* forward decl */

static PyObject *
Timer_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"interval", "function", "args", "kwargs", NULL};
    double interval;
    PyObject *function;
    PyObject *t_args = NULL;
    PyObject *t_kwargs = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dO|OO", kwlist,
                                     &interval, &function, &t_args, &t_kwargs))
        return NULL;

    if (!PyCallable_Check(function)) {
        PyErr_SetString(PyExc_TypeError, "function must be callable");
        return NULL;
    }

    TimerObject *self = (TimerObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    ct_mutex_init(&self->mu);
    ct_cond_init(&self->cv);

    Py_INCREF(function);
    self->function = function;
    self->interval = interval;

    if (t_args) { Py_INCREF(t_args); self->args = t_args; }
    else { self->args = PyTuple_New(0); if (!self->args) { Py_DECREF(self); return NULL; } }

    if (t_kwargs && t_kwargs != Py_None) { Py_INCREF(t_kwargs); self->kwargs = t_kwargs; }
    else { self->kwargs = NULL; }

    static atomic_ullong timer_counter = 0;
    unsigned long long n = atomic_fetch_add_explicit(&timer_counter, 1, memory_order_relaxed);
    self->name = PyUnicode_FromFormat("Timer-%llu", n);
    if (!self->name) { Py_DECREF(self); return NULL; }

    self->ident = 0;
    self->native_id = 0;
    self->started = 0;
    self->alive = 0;
    self->daemon = 1;   /* timers are daemon by default */
    atomic_init(&self->cancelled, 0);
    self->finished = 0;

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

    /* Wait for interval or cancellation */
    ct_mutex_lock(&self->mu);
    if (!atomic_load_explicit(&self->cancelled, memory_order_relaxed)) {
        unsigned long ms = (unsigned long)(self->interval * 1000.0);
        if (ms > 0)
            ct_cond_timedwait_ms(&self->cv, &self->mu, ms);
    }
    ct_mutex_unlock(&self->mu);

    /* Execute function if not cancelled */
    if (!atomic_load_explicit(&self->cancelled, memory_order_relaxed)) {
        PyObject *result;
        if (self->kwargs)
            result = PyObject_Call(self->function, self->args, self->kwargs);
        else
            result = PyObject_CallObject(self->function, self->args);
        Py_XDECREF(result);
        if (PyErr_Occurred())
            PyErr_Print();
    }

    ct_mutex_lock(&self->mu);
    self->alive = 0;
    self->finished = 1;
    ct_cond_broadcast(&self->cv);
    ct_mutex_unlock(&self->mu);

    registry_remove((PyObject *)self);
    Py_DECREF(self);

    PyGILState_Release(gstate);
}

static PyObject *
Timer_start(TimerObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->started) {
        PyErr_SetString(PyExc_RuntimeError, "threads can only be started once");
        return NULL;
    }
    self->started = 1;
    self->alive = 1;   /* set BEFORE spawning so join() always blocks */

    Py_INCREF(self);
    registry_add((PyObject *)self);

    unsigned long ident = PyThread_start_new_thread(_timer_bootstrap, (void *)self);
    if (ident == (unsigned long)-1) {
        self->started = 0;
        self->alive = 0;
        registry_remove((PyObject *)self);
        Py_DECREF(self);
        PyErr_SetString(PyExc_RuntimeError, "can't start new thread");
        return NULL;
    }
    self->ident = ident;
    Py_RETURN_NONE;
}

static PyObject *
Timer_cancel(TimerObject *self, PyObject *Py_UNUSED(ignored))
{
    atomic_store_explicit(&self->cancelled, 1, memory_order_relaxed);
    ct_mutex_lock(&self->mu);
    ct_cond_signal(&self->cv);
    ct_mutex_unlock(&self->mu);
    Py_RETURN_NONE;
}

static PyObject *
Timer_join(TimerObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"timeout", NULL};
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &timeout))
        return NULL;

    if (!self->started) {
        PyErr_SetString(PyExc_RuntimeError, "cannot join thread before it is started");
        return NULL;
    }

    ct_mutex_lock(&self->mu);
    if (timeout < 0) {
        while (self->alive)
            ct_cond_wait(&self->cv, &self->mu);
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
    ct_mutex_unlock(&self->mu);
    Py_RETURN_NONE;
}

static PyObject *
Timer_is_alive(TimerObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->alive) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Timer_get_name(TimerObject *self, void *Py_UNUSED(closure))
{
    Py_INCREF(self->name);
    return self->name;
}

static PyObject *
Timer_get_daemon(TimerObject *self, void *Py_UNUSED(closure))
{
    if (self->daemon) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyMethodDef Timer_methods[] = {
    {"start",    (PyCFunction)Timer_start,    METH_NOARGS,                  "Start the timer"},
    {"cancel",   (PyCFunction)Timer_cancel,   METH_NOARGS,                  "Cancel the timer"},
    {"join",     (PyCFunction)Timer_join,     METH_VARARGS | METH_KEYWORDS, "Wait for timer to finish"},
    {"is_alive", (PyCFunction)Timer_is_alive, METH_NOARGS,                  "Return whether the timer is alive"},
    {NULL}
};

static PyGetSetDef Timer_getset[] = {
    {"name",   (getter)Timer_get_name,   NULL, "Timer name", NULL},
    {"daemon", (getter)Timer_get_daemon, NULL, "Daemon flag", NULL},
    {NULL}
};

static PyTypeObject TimerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._threading.Timer",
    .tp_doc       = "C-backed timer thread — calls function after interval seconds.",
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
 * MODULE-LEVEL THREADING FUNCTIONS
 * ================================================================ */

static PyObject *
mod_active_count(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    ct_mutex_lock(&_registry.mu);
    int count = _registry.count + 1;  /* +1 for main thread */
    ct_mutex_unlock(&_registry.mu);
    return PyLong_FromLong(count);
}

static PyObject *
mod_current_thread(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    unsigned long tid = PyThread_get_thread_ident();

    /* Check if main thread */
    if (tid == _registry.main_tid && _registry.main_thread_obj) {
        Py_INCREF(_registry.main_thread_obj);
        return _registry.main_thread_obj;
    }

    /* Search registry */
    ct_mutex_lock(&_registry.mu);
    for (int i = 0; i < _registry.count; i++) {
        PyObject *t = _registry.threads[i];
        /* Check Thread type */
        if (Py_IS_TYPE(t, &ThreadType)) {
            if (((ThreadObject *)t)->ident == tid) {
                Py_INCREF(t);
                ct_mutex_unlock(&_registry.mu);
                return t;
            }
        } else if (Py_IS_TYPE(t, &TimerType)) {
            if (((TimerObject *)t)->ident == tid) {
                Py_INCREF(t);
                ct_mutex_unlock(&_registry.mu);
                return t;
            }
        }
    }
    ct_mutex_unlock(&_registry.mu);

    /* Not found — return main thread as fallback */
    if (_registry.main_thread_obj) {
        Py_INCREF(_registry.main_thread_obj);
        return _registry.main_thread_obj;
    }
    Py_RETURN_NONE;
}

static PyObject *
mod_main_thread(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    if (_registry.main_thread_obj) {
        Py_INCREF(_registry.main_thread_obj);
        return _registry.main_thread_obj;
    }
    Py_RETURN_NONE;
}

static PyObject *
mod_enumerate_threads(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    ct_mutex_lock(&_registry.mu);
    PyObject *list = PyList_New(_registry.count + 1);
    if (!list) { ct_mutex_unlock(&_registry.mu); return NULL; }

    /* Add main thread first */
    if (_registry.main_thread_obj) {
        Py_INCREF(_registry.main_thread_obj);
        PyList_SET_ITEM(list, 0, _registry.main_thread_obj);
    } else {
        Py_INCREF(Py_None);
        PyList_SET_ITEM(list, 0, Py_None);
    }

    for (int i = 0; i < _registry.count; i++) {
        Py_INCREF(_registry.threads[i]);
        PyList_SET_ITEM(list, i + 1, _registry.threads[i]);
    }
    ct_mutex_unlock(&_registry.mu);
    return list;
}

static PyObject *
mod_get_ident(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    return PyLong_FromUnsignedLong(PyThread_get_thread_ident());
}

static PyObject *
mod_get_native_id(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
#ifdef _WIN32
    return PyLong_FromUnsignedLong((unsigned long)GetCurrentThreadId());
#else
    return PyLong_FromUnsignedLong((unsigned long)pthread_self());
#endif
}

static PyObject *
mod_stack_size(PyObject *Py_UNUSED(self), PyObject *args)
{
    Py_ssize_t new_size = 0;
    if (!PyArg_ParseTuple(args, "|n", &new_size))
        return NULL;

    /* CPython uses PyThread_set_stacksize, we mirror that */
    size_t old_size = 0;
    if (new_size > 0) {
        /* Validate minimum */
        if (new_size < 32768) {
            PyErr_SetString(PyExc_ValueError, "size must be >= 32768 (32 KiB)");
            return NULL;
        }
        /* We don't actually change the stack size in our implementation,
           but we accept the call for API compatibility */
    }
    return PyLong_FromSsize_t((Py_ssize_t)old_size);
}

/* TIMEOUT_MAX: maximum timeout value (~292 years in seconds, same as CPython) */
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
    {"cpu_count",             (PyCFunction)mod_cpu_count,             METH_NOARGS,  "Return the number of CPU cores (logical)"},
    {"physical_cpu_count",    (PyCFunction)mod_physical_cpu_count,    METH_NOARGS,  "Return the number of physical CPU cores"},
    {"set_default_pool_size", (PyCFunction)mod_set_default_pool_size, METH_VARARGS, "Set the default pool worker count"},
    {"get_default_pool",      (PyCFunction)mod_get_default_pool,      METH_NOARGS,  "Get the default ThreadPool"},
    {"parallel_map",          (PyCFunction)mod_parallel_map,          METH_VARARGS | METH_KEYWORDS, "parallel_map(fn, items, num_workers=0) — work-stealing parallel map"},
    {"parallel_starmap",      (PyCFunction)mod_parallel_starmap,      METH_VARARGS | METH_KEYWORDS, "parallel_starmap(fn, items, num_workers=0) — work-stealing parallel starmap"},
    {"pool_map",              (PyCFunction)mod_pool_map,              METH_VARARGS | METH_KEYWORDS, "pool_map(fn, items, num_workers=0) — pool-based parallel map (no thread creation)"},
    {"pool_starmap",          (PyCFunction)mod_pool_starmap,          METH_VARARGS | METH_KEYWORDS, "pool_starmap(fn, items, num_workers=0) — pool-based parallel starmap (no thread creation)"},
    {"active_count",          (PyCFunction)mod_active_count,          METH_NOARGS,  "Return the number of alive threads"},
    {"current_thread",        (PyCFunction)mod_current_thread,        METH_NOARGS,  "Return the current Thread object"},
    {"main_thread",           (PyCFunction)mod_main_thread,           METH_NOARGS,  "Return the main Thread object"},
    {"enumerate",             (PyCFunction)mod_enumerate_threads,     METH_NOARGS,  "Return a list of all alive Thread objects"},
    {"get_ident",             (PyCFunction)mod_get_ident,             METH_NOARGS,  "Return the thread identifier of the current thread"},
    {"get_native_id",         (PyCFunction)mod_get_native_id,         METH_NOARGS,  "Return the native integral Thread ID"},
    {"stack_size",            (PyCFunction)mod_stack_size,            METH_VARARGS, "Return/set the thread stack size"},
    {NULL, NULL, 0, NULL},
};

static PyModuleDef threading_module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name    = "cthreading._threading",
    .m_doc     = "cthreading C core — ThreadPool, auto_thread, cpu_count",
    .m_size    = -1,
    .m_methods = threading_module_methods,
    .m_free    = (freefunc)threading_module_free,
};

PyMODINIT_FUNC
PyInit__threading(void)
{
    PyObject *m;

    if (PyType_Ready(&ThreadPoolType) < 0)  return NULL;
    if (PyType_Ready(&AutoThreadType) < 0)  return NULL;
    if (PyType_Ready(&ThreadType) < 0)      return NULL;
    if (PyType_Ready(&TimerType) < 0)       return NULL;

    m = PyModule_Create(&threading_module_def);
    if (m == NULL)
        return NULL;

    /* Initialize the thread registry */
    registry_init();

    /* Create main thread sentinel */
    {
        PyObject *empty_args = PyTuple_New(0);
        PyObject *kw = PyDict_New();
        if (!empty_args || !kw) { Py_XDECREF(empty_args); Py_XDECREF(kw); Py_DECREF(m); return NULL; }
        PyObject *main_name = PyUnicode_FromString("MainThread");
        if (!main_name) { Py_DECREF(empty_args); Py_DECREF(kw); Py_DECREF(m); return NULL; }
        PyDict_SetItemString(kw, "name", main_name);
        Py_DECREF(main_name);
        _registry.main_thread_obj = Thread_new(&ThreadType, empty_args, kw);
        Py_DECREF(empty_args);
        Py_DECREF(kw);
        if (!_registry.main_thread_obj) { Py_DECREF(m); return NULL; }
        ThreadObject *mt = (ThreadObject *)_registry.main_thread_obj;
        mt->started = 1;
        mt->alive = 1;
        mt->ident = _registry.main_tid;
#ifdef _WIN32
        mt->native_id = (unsigned long)GetCurrentThreadId();
#else
        mt->native_id = (unsigned long)pthread_self();
#endif
    }

#define ADD_TYPE(name, typeobj)                             \
    Py_INCREF(&(typeobj));                                 \
    if (PyModule_AddObject(m, name, (PyObject *)&(typeobj)) < 0) { \
        Py_DECREF(&(typeobj));                             \
        Py_DECREF(m);                                     \
        return NULL;                                       \
    }

    ADD_TYPE("ThreadPool",  ThreadPoolType);
    ADD_TYPE("auto_thread", AutoThreadType);
    ADD_TYPE("Thread",      ThreadType);
    ADD_TYPE("Timer",       TimerType);

#undef ADD_TYPE

    /* Add TIMEOUT_MAX constant */
    if (PyModule_AddObject(m, "TIMEOUT_MAX", PyFloat_FromDouble(CT_TIMEOUT_MAX)) < 0) {
        Py_DECREF(m);
        return NULL;
    }

#ifdef Py_MOD_GIL_NOT_USED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    return m;
}
