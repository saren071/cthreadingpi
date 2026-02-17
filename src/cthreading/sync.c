/*
 * src/cthreading/sync.c
 *
 * High-performance synchronization primitives implemented in C:
 *   - Lock        (non-reentrant mutex)
 *   - RLock       (reentrant mutex)
 *   - Event       (manual-reset event)
 *   - Semaphore   (counting semaphore)
 *   - Condition   (condition variable, wraps a Lock/RLock)
 *
 * All primitives support context-manager protocol and optional
 * contention telemetry when monitoring is enabled.
 */

#include "cthreading_common.h"

/* Per-module telemetry flag (separate DLL, can't share with monitoring) */
static atomic_int cthreading_telemetry_enabled;

/* ================================================================
 * LOCK (non-reentrant) — mutex + condvar for full timeout support
 * Fast CAS path for uncontended case, condvar slow path for waits.
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    ct_mutex_t         mu;
    ct_cond_t          cv;
    atomic_int         held;
    atomic_ullong      access_count;
    atomic_ullong      contention_count;
} LockObject;

static void
Lock_dealloc(LockObject *self)
{
    ct_cond_destroy(&self->cv);
    ct_mutex_destroy(&self->mu);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Lock_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    LockObject *self = (LockObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;
    ct_mutex_init(&self->mu);
    ct_cond_init(&self->cv);
    atomic_init(&self->held, 0);
    atomic_init(&self->access_count, 0);
    atomic_init(&self->contention_count, 0);
    return (PyObject *)self;
}

static PyObject *
Lock_acquire(LockObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"blocking", "timeout", NULL};
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pd", kwlist, &blocking, &timeout))
        return NULL;

    /* Fast path: CAS held 0→1 */
    int expected = 0;
    if (atomic_compare_exchange_strong_explicit(&self->held, &expected, 1,
            memory_order_acquire, memory_order_relaxed)) {
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
        Py_RETURN_TRUE;
    }

    if (!blocking)
        Py_RETURN_FALSE;

    /* Contention */
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->contention_count, 1, memory_order_relaxed);

    /* Slow path: mutex + condvar */
    ct_mutex_lock(&self->mu);
    if (timeout < 0) {
        /* Infinite wait */
        while (atomic_load_explicit(&self->held, memory_order_relaxed))
            ct_cond_wait(&self->cv, &self->mu);
        atomic_store_explicit(&self->held, 1, memory_order_release);
        ct_mutex_unlock(&self->mu);
    } else {
        /* Timed wait */
        double deadline = ct_time_ms() + timeout * 1000.0;
        while (atomic_load_explicit(&self->held, memory_order_relaxed)) {
            double remaining = deadline - ct_time_ms();
            if (remaining <= 0) {
                ct_mutex_unlock(&self->mu);
                Py_RETURN_FALSE;
            }
            unsigned long ms = (unsigned long)remaining;
            if (ms == 0) ms = 1;
            ct_cond_timedwait_ms(&self->cv, &self->mu, ms);
        }
        atomic_store_explicit(&self->held, 1, memory_order_release);
        ct_mutex_unlock(&self->mu);
    }

    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
    Py_RETURN_TRUE;
}

static PyObject *
Lock_release(LockObject *self, PyObject *Py_UNUSED(ignored))
{
    if (!atomic_load_explicit(&self->held, memory_order_relaxed)) {
        PyErr_SetString(PyExc_RuntimeError, "release unlocked lock");
        return NULL;
    }
    ct_mutex_lock(&self->mu);
    atomic_store_explicit(&self->held, 0, memory_order_release);
    ct_cond_signal(&self->cv);
    ct_mutex_unlock(&self->mu);
    Py_RETURN_NONE;
}

static PyObject *
Lock_locked(LockObject *self, PyObject *Py_UNUSED(ignored))
{
    if (atomic_load_explicit(&self->held, memory_order_relaxed))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Lock_enter(LockObject *self, PyObject *Py_UNUSED(ignored))
{
    /* Fast path: CAS */
    int expected = 0;
    if (atomic_compare_exchange_strong_explicit(&self->held, &expected, 1,
            memory_order_acquire, memory_order_relaxed)) {
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
        Py_INCREF(self);
        return (PyObject *)self;
    }
    /* Contention */
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->contention_count, 1, memory_order_relaxed);
    ct_mutex_lock(&self->mu);
    while (atomic_load_explicit(&self->held, memory_order_relaxed))
        ct_cond_wait(&self->cv, &self->mu);
    atomic_store_explicit(&self->held, 1, memory_order_release);
    ct_mutex_unlock(&self->mu);
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
Lock_exit(LockObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    ct_mutex_lock(&self->mu);
    atomic_store_explicit(&self->held, 0, memory_order_release);
    ct_cond_signal(&self->cv);
    ct_mutex_unlock(&self->mu);
    Py_RETURN_NONE;
}

static PyObject *
Lock_stats(LockObject *self, PyObject *Py_UNUSED(ignored))
{
    return Py_BuildValue("{s:K, s:K}",
        "accesses",  atomic_load_explicit(&self->access_count, memory_order_relaxed),
        "contention", atomic_load_explicit(&self->contention_count, memory_order_relaxed));
}

static PyMethodDef Lock_methods[] = {
    {"acquire",    (PyCFunction)Lock_acquire, METH_VARARGS | METH_KEYWORDS, "Acquire the lock"},
    {"release",    (PyCFunction)Lock_release, METH_NOARGS,  "Release the lock"},
    {"locked",     (PyCFunction)Lock_locked,  METH_NOARGS,  "Return whether the lock is held"},
    {"__enter__",  (PyCFunction)Lock_enter,   METH_NOARGS,  "Acquire (context manager)"},
    {"__exit__",   (PyCFunction)Lock_exit,    METH_FASTCALL, "Release (context manager)"},
    {"stats",      (PyCFunction)Lock_stats,   METH_NOARGS,  "Get contention telemetry"},
    {NULL}
};

static PyTypeObject LockType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._sync.Lock",
    .tp_doc       = "High-performance non-reentrant lock with full timeout support.",
    .tp_basicsize = sizeof(LockObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new       = Lock_new,
    .tp_dealloc   = (destructor)Lock_dealloc,
    .tp_methods   = Lock_methods,
};

/* ================================================================
 * RLOCK (reentrant) — mutex + condvar for full timeout support
 * Fast owner-check for reentrant case, condvar for contended waits.
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    ct_mutex_t         mu;
    ct_cond_t          cv;
    atomic_ulong       owner;
    unsigned int       recursion;
    atomic_ullong      access_count;
    atomic_ullong      contention_count;
} RLockObject;

static void
RLock_dealloc(RLockObject *self)
{
    ct_cond_destroy(&self->cv);
    ct_mutex_destroy(&self->mu);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
RLock_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    RLockObject *self = (RLockObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;
    ct_mutex_init(&self->mu);
    ct_cond_init(&self->cv);
    atomic_init(&self->owner, 0);
    self->recursion = 0;
    atomic_init(&self->access_count, 0);
    atomic_init(&self->contention_count, 0);
    return (PyObject *)self;
}

static PyObject *
RLock_acquire(RLockObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"blocking", "timeout", NULL};
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pd", kwlist, &blocking, &timeout))
        return NULL;

    unsigned long tid = PyThread_get_thread_ident();

    /* Fast path: reentrant acquire by same thread */
    if (atomic_load_explicit(&self->owner, memory_order_relaxed) == tid) {
        self->recursion++;
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
        Py_RETURN_TRUE;
    }

    ct_mutex_lock(&self->mu);

    /* Try immediate acquire */
    if (atomic_load_explicit(&self->owner, memory_order_relaxed) == 0) {
        atomic_store_explicit(&self->owner, tid, memory_order_relaxed);
        self->recursion = 1;
        ct_mutex_unlock(&self->mu);
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
        Py_RETURN_TRUE;
    }

    if (!blocking) {
        ct_mutex_unlock(&self->mu);
        Py_RETURN_FALSE;
    }

    /* Contention */
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->contention_count, 1, memory_order_relaxed);

    if (timeout < 0) {
        /* Infinite wait */
        while (atomic_load_explicit(&self->owner, memory_order_relaxed) != 0)
            ct_cond_wait(&self->cv, &self->mu);
    } else {
        /* Timed wait */
        double deadline = ct_time_ms() + timeout * 1000.0;
        while (atomic_load_explicit(&self->owner, memory_order_relaxed) != 0) {
            double remaining = deadline - ct_time_ms();
            if (remaining <= 0) {
                ct_mutex_unlock(&self->mu);
                Py_RETURN_FALSE;
            }
            unsigned long ms = (unsigned long)remaining;
            if (ms == 0) ms = 1;
            ct_cond_timedwait_ms(&self->cv, &self->mu, ms);
        }
    }

    atomic_store_explicit(&self->owner, tid, memory_order_relaxed);
    self->recursion = 1;
    ct_mutex_unlock(&self->mu);

    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
    Py_RETURN_TRUE;
}

static PyObject *
RLock_release(RLockObject *self, PyObject *Py_UNUSED(ignored))
{
    unsigned long tid = PyThread_get_thread_ident();
    if (atomic_load_explicit(&self->owner, memory_order_relaxed) != tid) {
        PyErr_SetString(PyExc_RuntimeError, "cannot release un-acquired lock");
        return NULL;
    }
    if (self->recursion > 1) {
        self->recursion--;
        Py_RETURN_NONE;
    }
    ct_mutex_lock(&self->mu);
    self->recursion = 0;
    atomic_store_explicit(&self->owner, 0, memory_order_relaxed);
    ct_cond_signal(&self->cv);
    ct_mutex_unlock(&self->mu);
    Py_RETURN_NONE;
}

static PyObject *
RLock_enter(RLockObject *self, PyObject *Py_UNUSED(ignored))
{
    unsigned long tid = PyThread_get_thread_ident();

    /* Fast path: reentrant */
    if (atomic_load_explicit(&self->owner, memory_order_relaxed) == tid) {
        self->recursion++;
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
        Py_INCREF(self);
        return (PyObject *)self;
    }

    ct_mutex_lock(&self->mu);
    if (atomic_load_explicit(&self->owner, memory_order_relaxed) != 0) {
        /* Contention */
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->contention_count, 1, memory_order_relaxed);
        while (atomic_load_explicit(&self->owner, memory_order_relaxed) != 0)
            ct_cond_wait(&self->cv, &self->mu);
    }
    atomic_store_explicit(&self->owner, tid, memory_order_relaxed);
    self->recursion = 1;
    ct_mutex_unlock(&self->mu);

    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
RLock_exit(RLockObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return RLock_release(self, NULL);
}

static PyObject *
RLock_stats(RLockObject *self, PyObject *Py_UNUSED(ignored))
{
    return Py_BuildValue("{s:K, s:K}",
        "accesses",   atomic_load_explicit(&self->access_count, memory_order_relaxed),
        "contention", atomic_load_explicit(&self->contention_count, memory_order_relaxed));
}

static PyMethodDef RLock_methods[] = {
    {"acquire",    (PyCFunction)RLock_acquire, METH_VARARGS | METH_KEYWORDS, "Acquire the lock"},
    {"release",    (PyCFunction)RLock_release, METH_NOARGS,  "Release the lock"},
    {"__enter__",  (PyCFunction)RLock_enter,   METH_NOARGS,  "Acquire (context manager)"},
    {"__exit__",   (PyCFunction)RLock_exit,    METH_FASTCALL, "Release (context manager)"},
    {"stats",      (PyCFunction)RLock_stats,   METH_NOARGS,  "Get contention telemetry"},
    {NULL}
};

static PyTypeObject RLockType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._sync.RLock",
    .tp_doc       = "High-performance reentrant lock with full timeout support.",
    .tp_basicsize = sizeof(RLockObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new       = RLock_new,
    .tp_dealloc   = (destructor)RLock_dealloc,
    .tp_methods   = RLock_methods,
};

/* ================================================================
 * EVENT (manual-reset) — ct_mutex + ct_cond (no kernel objects)
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    ct_mutex_t    lock;
    ct_cond_t     cond;
    atomic_int    flag;
} EventObject;

static void
Event_dealloc(EventObject *self)
{
    ct_cond_destroy(&self->cond);
    ct_mutex_destroy(&self->lock);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Event_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    EventObject *self = (EventObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;
    ct_mutex_init(&self->lock);
    ct_cond_init(&self->cond);
    atomic_init(&self->flag, 0);
    return (PyObject *)self;
}

static PyObject *
Event_is_set(EventObject *self, PyObject *Py_UNUSED(ignored))
{
    if (atomic_load_explicit(&self->flag, memory_order_acquire))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Event_set(EventObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->lock);
    atomic_store_explicit(&self->flag, 1, memory_order_release);
    ct_cond_broadcast(&self->cond);
    ct_mutex_unlock(&self->lock);
    Py_RETURN_NONE;
}

static PyObject *
Event_clear(EventObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->lock);
    atomic_store_explicit(&self->flag, 0, memory_order_release);
    ct_mutex_unlock(&self->lock);
    Py_RETURN_NONE;
}

static PyObject *
Event_wait(EventObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"timeout", NULL};
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &timeout))
        return NULL;

    if (atomic_load_explicit(&self->flag, memory_order_acquire))
        Py_RETURN_TRUE;

    ct_mutex_lock(&self->lock);
    if (timeout < 0) {
        while (!atomic_load_explicit(&self->flag, memory_order_acquire))
            ct_cond_wait(&self->cond, &self->lock);
    } else {
        DWORD ms = (DWORD)(timeout * 1000.0);
        if (!atomic_load_explicit(&self->flag, memory_order_acquire))
            ct_cond_timedwait_ms(&self->cond, &self->lock, ms);
    }
    ct_mutex_unlock(&self->lock);

    if (atomic_load_explicit(&self->flag, memory_order_acquire))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Event_check(EventObject *self, PyObject *Py_UNUSED(ignored))
{
    if (atomic_load_explicit(&self->flag, memory_order_acquire))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyMethodDef Event_methods[] = {
    {"is_set", (PyCFunction)Event_is_set, METH_NOARGS,                   "Return True if set"},
    {"set",    (PyCFunction)Event_set,    METH_NOARGS,                   "Set the event flag"},
    {"clear",  (PyCFunction)Event_clear,  METH_NOARGS,                   "Clear the event flag"},
    {"wait",   (PyCFunction)Event_wait,   METH_VARARGS | METH_KEYWORDS,  "Block until set or timeout"},
    {"check",  (PyCFunction)Event_check,  METH_NOARGS,                   "Fast non-blocking check if set (no arg parsing)"},
    {NULL}
};

static PyTypeObject EventType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._sync.Event",
    .tp_doc       = "High-performance manual-reset event (condvar-based).",
    .tp_basicsize = sizeof(EventObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new       = Event_new,
    .tp_dealloc   = (destructor)Event_dealloc,
    .tp_methods   = Event_methods,
};

/* ================================================================
 * SEMAPHORE (counting) — ct_mutex + ct_cond (no kernel objects)
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    ct_mutex_t    lock;
    ct_cond_t     cond;
    atomic_int    count;
    int           max_count;
    atomic_int    waiters;          /* track waiting threads to skip signal */
    atomic_ullong access_count;
    atomic_ullong contention_count;
} SemaphoreObject;

static void
Semaphore_dealloc(SemaphoreObject *self)
{
    ct_cond_destroy(&self->cond);
    ct_mutex_destroy(&self->lock);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Semaphore_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"value", "max_value", NULL};
    int value = 1;
    int max_value = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ii", kwlist, &value, &max_value))
        return NULL;
    if (value < 0) {
        PyErr_SetString(PyExc_ValueError, "semaphore initial value must be >= 0");
        return NULL;
    }

    SemaphoreObject *self = (SemaphoreObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    ct_mutex_init(&self->lock);
    ct_cond_init(&self->cond);
    atomic_init(&self->count, value);
    self->max_count = max_value;
    atomic_init(&self->waiters, 0);
    atomic_init(&self->access_count, 0);
    atomic_init(&self->contention_count, 0);

    return (PyObject *)self;
}

/* Try atomic CAS decrement: returns 1 on success, 0 on failure */
static inline int
Semaphore_try_dec(SemaphoreObject *self)
{
    int cur = atomic_load_explicit(&self->count, memory_order_relaxed);
    while (cur > 0) {
        if (atomic_compare_exchange_weak_explicit(
                &self->count, &cur, cur - 1,
                memory_order_acquire, memory_order_relaxed))
            return 1;
    }
    return 0;
}

static PyObject *
Semaphore_acquire(SemaphoreObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"blocking", "timeout", NULL};
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pd", kwlist, &blocking, &timeout))
        return NULL;

    /* FAST PATH: atomic CAS — no mutex at all */
    if (Semaphore_try_dec(self)) {
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
        Py_RETURN_TRUE;
    }

    if (!blocking)
        Py_RETURN_FALSE;

    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->contention_count, 1, memory_order_relaxed);

    /* SLOW PATH: condvar wait */
    ct_mutex_lock(&self->lock);
    atomic_fetch_add_explicit(&self->waiters, 1, memory_order_relaxed);
    if (timeout < 0) {
        while (atomic_load_explicit(&self->count, memory_order_relaxed) <= 0)
            ct_cond_wait(&self->cond, &self->lock);
    } else {
        DWORD ms = (DWORD)(timeout * 1000.0);
        if (atomic_load_explicit(&self->count, memory_order_relaxed) <= 0)
            ct_cond_timedwait_ms(&self->cond, &self->lock, ms);
    }
    atomic_fetch_sub_explicit(&self->waiters, 1, memory_order_relaxed);
    int success = Semaphore_try_dec(self);
    ct_mutex_unlock(&self->lock);

    if (success) {
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyObject *
Semaphore_release(SemaphoreObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"n", NULL};
    int n = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &n))
        return NULL;

    if (self->max_count > 0) {
        int cur = atomic_load_explicit(&self->count, memory_order_relaxed);
        if (cur + n > self->max_count) {
            PyErr_SetString(PyExc_ValueError, "Semaphore released too many times");
            return NULL;
        }
    }
    atomic_fetch_add_explicit(&self->count, n, memory_order_release);
    /* Wake waiters via condvar — skip if no one is waiting */
    if (atomic_load_explicit(&self->waiters, memory_order_relaxed) > 0) {
        ct_mutex_lock(&self->lock);
        for (int i = 0; i < n; i++)
            ct_cond_signal(&self->cond);
        ct_mutex_unlock(&self->lock);
    }
    Py_RETURN_NONE;
}

static PyObject *
Semaphore_enter(SemaphoreObject *self, PyObject *Py_UNUSED(ignored))
{
    /* Fast CAS path */
    if (Semaphore_try_dec(self)) {
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
        Py_INCREF(self);
        return (PyObject *)self;
    }
    /* Slow condvar path */
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->contention_count, 1, memory_order_relaxed);
    ct_mutex_lock(&self->lock);
    atomic_fetch_add_explicit(&self->waiters, 1, memory_order_relaxed);
    while (atomic_load_explicit(&self->count, memory_order_relaxed) <= 0)
        ct_cond_wait(&self->cond, &self->lock);
    atomic_fetch_sub_explicit(&self->waiters, 1, memory_order_relaxed);
    Semaphore_try_dec(self);
    ct_mutex_unlock(&self->lock);
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
Semaphore_exit(SemaphoreObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    atomic_fetch_add_explicit(&self->count, 1, memory_order_release);
    /* Only signal if there are actual waiters */
    if (atomic_load_explicit(&self->waiters, memory_order_relaxed) > 0) {
        ct_mutex_lock(&self->lock);
        ct_cond_signal(&self->cond);
        ct_mutex_unlock(&self->lock);
    }
    Py_RETURN_NONE;
}

static PyObject *
Semaphore_stats(SemaphoreObject *self, PyObject *Py_UNUSED(ignored))
{
    return Py_BuildValue("{s:K, s:K}",
        "accesses",   atomic_load_explicit(&self->access_count, memory_order_relaxed),
        "contention", atomic_load_explicit(&self->contention_count, memory_order_relaxed));
}

static PyMethodDef Semaphore_methods[] = {
    {"acquire",   (PyCFunction)Semaphore_acquire, METH_VARARGS | METH_KEYWORDS, "Acquire the semaphore"},
    {"release",   (PyCFunction)Semaphore_release, METH_VARARGS | METH_KEYWORDS, "Release the semaphore"},
    {"__enter__", (PyCFunction)Semaphore_enter,   METH_NOARGS,                  "Acquire (context manager)"},
    {"__exit__",  (PyCFunction)Semaphore_exit,    METH_FASTCALL,                "Release (context manager)"},
    {"stats",     (PyCFunction)Semaphore_stats,   METH_NOARGS,                  "Get contention telemetry"},
    {NULL}
};

static PyTypeObject SemaphoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._sync.Semaphore",
    .tp_doc       = "High-performance counting semaphore (condvar-based).",
    .tp_basicsize = sizeof(SemaphoreObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new       = Semaphore_new,
    .tp_dealloc   = (destructor)Semaphore_dealloc,
    .tp_methods   = Semaphore_methods,
};

/* ================================================================
 * CONDITION VARIABLE — native ct_mutex + ct_cond
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    PyObject    *py_lock;       /* the Lock or RLock being wrapped */
    ct_mutex_t   cond_mutex;
    ct_cond_t    cond;
    int          notify_count;  /* how many waiters to wake         */
    unsigned int generation;    /* incremented by notify_all         */
    int          is_c_lock;     /* 1=Lock, 2=RLock, 0=unknown       */
} ConditionObject;

static int
Condition_traverse(ConditionObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->py_lock);
    return 0;
}

static int
Condition_clear_refs(ConditionObject *self)
{
    Py_CLEAR(self->py_lock);
    return 0;
}

static void
Condition_dealloc(ConditionObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_XDECREF(self->py_lock);
    ct_cond_destroy(&self->cond);
    ct_mutex_destroy(&self->cond_mutex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Condition_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"lock", NULL};
    PyObject *lock_obj = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &lock_obj))
        return NULL;

    ConditionObject *self = (ConditionObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    if (lock_obj == NULL || lock_obj == Py_None) {
        PyObject *empty = PyTuple_New(0);
        if (empty == NULL) { Py_DECREF(self); return NULL; }
        lock_obj = RLock_new(&RLockType, empty, NULL);
        Py_DECREF(empty);
        if (lock_obj == NULL) { Py_DECREF(self); return NULL; }
    } else {
        Py_INCREF(lock_obj);
    }
    self->py_lock = lock_obj;
    ct_mutex_init(&self->cond_mutex);
    ct_cond_init(&self->cond);
    self->notify_count = 0;
    self->generation = 0;

    /* Detect C-backed lock type for direct dispatch */
    if (Py_IS_TYPE(lock_obj, &LockType))
        self->is_c_lock = 1;
    else if (Py_IS_TYPE(lock_obj, &RLockType))
        self->is_c_lock = 2;
    else
        self->is_c_lock = 0;

    return (PyObject *)self;
}

/* Direct C dispatch helpers for known lock types */
static inline PyObject *
_Condition_lock_enter(ConditionObject *self)
{
    if (self->is_c_lock == 1)
        return Lock_enter((LockObject *)self->py_lock, NULL);
    else if (self->is_c_lock == 2)
        return RLock_enter((RLockObject *)self->py_lock, NULL);
    else
        return PyObject_CallMethod(self->py_lock, "__enter__", NULL);
}

static inline PyObject *
_Condition_lock_exit(ConditionObject *self)
{
    if (self->is_c_lock == 1) {
        return Lock_exit((LockObject *)self->py_lock, NULL, 0);
    } else if (self->is_c_lock == 2) {
        return RLock_exit((RLockObject *)self->py_lock, NULL, 0);
    } else {
        return PyObject_CallMethod(self->py_lock, "__exit__", "OOO",
            Py_None, Py_None, Py_None);
    }
}

static inline PyObject *
_Condition_lock_acquire(ConditionObject *self, PyObject *args, PyObject *kwds)
{
    if (self->is_c_lock == 1)
        return Lock_acquire((LockObject *)self->py_lock, args, kwds);
    else if (self->is_c_lock == 2)
        return RLock_acquire((RLockObject *)self->py_lock, args, kwds);
    else {
        PyObject *method = PyObject_GetAttrString(self->py_lock, "acquire");
        if (!method) return NULL;
        PyObject *r = PyObject_Call(method, args, kwds);
        Py_DECREF(method);
        return r;
    }
}

static inline PyObject *
_Condition_lock_release(ConditionObject *self)
{
    if (self->is_c_lock == 1)
        return Lock_release((LockObject *)self->py_lock, NULL);
    else if (self->is_c_lock == 2)
        return RLock_release((RLockObject *)self->py_lock, NULL);
    else
        return PyObject_CallMethod(self->py_lock, "release", NULL);
}

static PyObject *
Condition_enter(ConditionObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *r = _Condition_lock_enter(self);
    if (r == NULL)
        return NULL;
    Py_DECREF(r);
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
Condition_exit(ConditionObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    return _Condition_lock_exit(self);
}

static PyObject *
Condition_acquire(ConditionObject *self, PyObject *args, PyObject *kwds)
{
    return _Condition_lock_acquire(self, args, kwds);
}

static PyObject *
Condition_release(ConditionObject *self, PyObject *Py_UNUSED(ignored))
{
    return _Condition_lock_release(self);
}

static PyObject *
Condition_wait(ConditionObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"timeout", NULL};
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &timeout))
        return NULL;

    /* Release the underlying lock (direct C dispatch if possible) */
    PyObject *r = _Condition_lock_release(self);
    if (r == NULL)
        return NULL;
    Py_DECREF(r);

    /* Wait on the native condvar */
    ct_mutex_lock(&self->cond_mutex);
    unsigned int my_gen = self->generation;
    if (timeout < 0) {
        while (self->notify_count <= 0 && self->generation == my_gen)
            ct_cond_wait(&self->cond, &self->cond_mutex);
    } else {
        double deadline = ct_time_ms() + timeout * 1000.0;
        while (self->notify_count <= 0 && self->generation == my_gen) {
            double remaining = deadline - ct_time_ms();
            if (remaining <= 0) break;
            unsigned long ms = (unsigned long)remaining;
            if (ms == 0) ms = 1;
            ct_cond_timedwait_ms(&self->cond, &self->cond_mutex, ms);
        }
    }
    int notified = (self->notify_count > 0 || self->generation != my_gen);
    if (self->notify_count > 0)
        self->notify_count--;
    ct_mutex_unlock(&self->cond_mutex);

    /* Reacquire the underlying lock (direct C dispatch if possible) */
    PyObject *acq_args = PyTuple_New(0);
    if (acq_args == NULL) return NULL;
    PyObject *acq_r = _Condition_lock_acquire(self, acq_args, NULL);
    Py_DECREF(acq_args);
    if (acq_r == NULL) return NULL;
    Py_DECREF(acq_r);

    if (notified)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Condition_notify(ConditionObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"n", NULL};
    int n = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &n))
        return NULL;

    ct_mutex_lock(&self->cond_mutex);
    self->notify_count += n;
    for (int i = 0; i < n; i++)
        ct_cond_signal(&self->cond);
    ct_mutex_unlock(&self->cond_mutex);
    Py_RETURN_NONE;
}

static PyObject *
Condition_notify_all(ConditionObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->cond_mutex);
    self->generation++;
    ct_cond_broadcast(&self->cond);
    ct_mutex_unlock(&self->cond_mutex);
    Py_RETURN_NONE;
}

static PyObject *
Condition_wait_for(ConditionObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"predicate", "timeout", NULL};
    PyObject *predicate;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|d", kwlist, &predicate, &timeout))
        return NULL;

    if (!PyCallable_Check(predicate)) {
        PyErr_SetString(PyExc_TypeError, "predicate must be callable");
        return NULL;
    }

    double deadline = -1.0;
    if (timeout >= 0)
        deadline = ct_time_ms() + timeout * 1000.0;

    for (;;) {
        PyObject *result = PyObject_CallNoArgs(predicate);
        if (!result) return NULL;
        int truthy = PyObject_IsTrue(result);
        Py_DECREF(result);
        if (truthy < 0) return NULL;
        if (truthy) Py_RETURN_TRUE;

        double wait_timeout = -1.0;
        if (deadline >= 0) {
            wait_timeout = (deadline - ct_time_ms()) / 1000.0;
            if (wait_timeout <= 0) Py_RETURN_FALSE;
        }

        /* Call Condition_wait with computed timeout */
        PyObject *wait_args;
        PyObject *wait_kwds = NULL;
        if (wait_timeout >= 0) {
            wait_args = PyTuple_New(0);
            if (!wait_args) return NULL;
            wait_kwds = PyDict_New();
            if (!wait_kwds) { Py_DECREF(wait_args); return NULL; }
            PyObject *to = PyFloat_FromDouble(wait_timeout);
            if (!to) { Py_DECREF(wait_args); Py_DECREF(wait_kwds); return NULL; }
            PyDict_SetItemString(wait_kwds, "timeout", to);
            Py_DECREF(to);
        } else {
            wait_args = PyTuple_New(0);
            if (!wait_args) return NULL;
        }
        PyObject *wait_r = Condition_wait(self, wait_args, wait_kwds);
        Py_DECREF(wait_args);
        Py_XDECREF(wait_kwds);
        if (!wait_r) return NULL;
        Py_DECREF(wait_r);
    }
}

static PyMethodDef Condition_methods[] = {
    {"acquire",    (PyCFunction)Condition_acquire,    METH_VARARGS | METH_KEYWORDS, "Acquire underlying lock"},
    {"release",    (PyCFunction)Condition_release,    METH_NOARGS,                  "Release underlying lock"},
    {"wait",       (PyCFunction)Condition_wait,       METH_VARARGS | METH_KEYWORDS, "Wait for notification"},
    {"wait_for",   (PyCFunction)Condition_wait_for,   METH_VARARGS | METH_KEYWORDS, "Wait until predicate is true"},
    {"notify",     (PyCFunction)Condition_notify,     METH_VARARGS | METH_KEYWORDS, "Notify n waiters"},
    {"notify_all", (PyCFunction)Condition_notify_all, METH_NOARGS,                  "Notify all waiters"},
    {"__enter__",  (PyCFunction)Condition_enter,      METH_NOARGS,                  "Acquire (context manager)"},
    {"__exit__",   (PyCFunction)Condition_exit,       METH_FASTCALL,                "Release (context manager)"},
    {NULL}
};

static PyTypeObject ConditionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._sync.Condition",
    .tp_doc       = "Condition variable backed by a C lock.",
    .tp_basicsize = sizeof(ConditionObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = Condition_new,
    .tp_dealloc   = (destructor)Condition_dealloc,
    .tp_traverse  = (traverseproc)Condition_traverse,
    .tp_clear     = (inquiry)Condition_clear_refs,
    .tp_methods   = Condition_methods,
};

/* ================================================================
 * BOUNDED SEMAPHORE — Semaphore that enforces max on release
 * ================================================================ */

static PyObject *
BoundedSemaphore_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"value", NULL};
    int value = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &value))
        return NULL;
    if (value < 0) {
        PyErr_SetString(PyExc_ValueError, "semaphore initial value must be >= 0");
        return NULL;
    }

    /* Delegate to Semaphore_new with max_value = value */
    PyObject *sem_args = Py_BuildValue("(ii)", value, value);
    if (!sem_args) return NULL;
    PyObject *result = Semaphore_new(&SemaphoreType, sem_args, NULL);
    Py_DECREF(sem_args);
    if (result) {
        /* Re-type to BoundedSemaphore */
        Py_SET_TYPE(result, type);
    }
    return result;
}

static PyTypeObject BoundedSemaphoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._sync.BoundedSemaphore",
    .tp_doc       = "Bounded semaphore — raises ValueError on over-release.",
    .tp_basicsize = sizeof(SemaphoreObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new       = BoundedSemaphore_new,
    .tp_dealloc   = (destructor)Semaphore_dealloc,
    .tp_methods   = Semaphore_methods,
};

/* ================================================================
 * BARRIER — multi-party synchronization primitive
 * ================================================================ */

static PyObject *BrokenBarrierError;

typedef struct {
    PyObject_HEAD
    ct_mutex_t      mu;
    ct_cond_t       cv;
    int             parties;
    int             waiting;
    unsigned int    generation;
    int             broken;
    PyObject       *action;
    double          default_timeout;
} BarrierObject;

static int
Barrier_traverse(BarrierObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->action);
    return 0;
}

static int
Barrier_clear_refs(BarrierObject *self)
{
    Py_CLEAR(self->action);
    return 0;
}

static void
Barrier_dealloc(BarrierObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_XDECREF(self->action);
    ct_cond_destroy(&self->cv);
    ct_mutex_destroy(&self->mu);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Barrier_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"parties", "action", "timeout", NULL};
    int parties;
    PyObject *action = NULL;
    double timeout = -1.0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|Od", kwlist,
                                     &parties, &action, &timeout))
        return NULL;
    if (parties < 1) {
        PyErr_SetString(PyExc_ValueError, "parties must be >= 1");
        return NULL;
    }

    BarrierObject *self = (BarrierObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    ct_mutex_init(&self->mu);
    ct_cond_init(&self->cv);
    self->parties = parties;
    self->waiting = 0;
    self->generation = 0;
    self->broken = 0;
    self->default_timeout = timeout;

    if (action && action != Py_None) {
        Py_INCREF(action);
        self->action = action;
    } else {
        self->action = NULL;
    }

    return (PyObject *)self;
}

static PyObject *
Barrier_wait(BarrierObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"timeout", NULL};
    double timeout = -999.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &timeout))
        return NULL;
    if (timeout == -999.0)
        timeout = self->default_timeout;

    ct_mutex_lock(&self->mu);

    if (self->broken) {
        ct_mutex_unlock(&self->mu);
        PyErr_SetObject(BrokenBarrierError,
            PyUnicode_FromString("barrier is broken"));
        return NULL;
    }

    unsigned int my_gen = self->generation;
    int index = self->waiting;
    self->waiting++;

    if (self->waiting == self->parties) {
        /* Last thread: run action, release everyone */
        self->waiting = 0;
        self->generation++;

        if (self->action) {
            PyObject *r = PyObject_CallNoArgs(self->action);
            if (!r) {
                self->broken = 1;
                ct_cond_broadcast(&self->cv);
                ct_mutex_unlock(&self->mu);
                return NULL;
            }
            Py_DECREF(r);
        }

        ct_cond_broadcast(&self->cv);
        ct_mutex_unlock(&self->mu);
        return PyLong_FromLong(index);
    }

    /* Wait for the barrier to be tripped */
    if (timeout < 0) {
        while (self->generation == my_gen && !self->broken)
            ct_cond_wait(&self->cv, &self->mu);
    } else {
        double deadline = ct_time_ms() + timeout * 1000.0;
        while (self->generation == my_gen && !self->broken) {
            double remaining = deadline - ct_time_ms();
            if (remaining <= 0) {
                /* Timeout: break the barrier */
                self->broken = 1;
                ct_cond_broadcast(&self->cv);
                ct_mutex_unlock(&self->mu);
                PyErr_SetObject(BrokenBarrierError,
                    PyUnicode_FromString("barrier timed out"));
                return NULL;
            }
            unsigned long ms = (unsigned long)remaining;
            if (ms == 0) ms = 1;
            ct_cond_timedwait_ms(&self->cv, &self->mu, ms);
        }
    }

    if (self->broken) {
        ct_mutex_unlock(&self->mu);
        PyErr_SetObject(BrokenBarrierError,
            PyUnicode_FromString("barrier is broken"));
        return NULL;
    }

    ct_mutex_unlock(&self->mu);
    return PyLong_FromLong(index);
}

static PyObject *
Barrier_reset(BarrierObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mu);
    if (self->waiting > 0) {
        self->broken = 1;
        ct_cond_broadcast(&self->cv);
    }
    self->waiting = 0;
    self->generation++;
    self->broken = 0;
    ct_mutex_unlock(&self->mu);
    Py_RETURN_NONE;
}

static PyObject *
Barrier_abort(BarrierObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mu);
    self->broken = 1;
    ct_cond_broadcast(&self->cv);
    ct_mutex_unlock(&self->mu);
    Py_RETURN_NONE;
}

static PyObject *
Barrier_get_parties(BarrierObject *self, void *Py_UNUSED(closure))
{
    return PyLong_FromLong(self->parties);
}

static PyObject *
Barrier_get_n_waiting(BarrierObject *self, void *Py_UNUSED(closure))
{
    ct_mutex_lock(&self->mu);
    int n = self->waiting;
    ct_mutex_unlock(&self->mu);
    return PyLong_FromLong(n);
}

static PyObject *
Barrier_get_broken(BarrierObject *self, void *Py_UNUSED(closure))
{
    if (self->broken)
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyMethodDef Barrier_methods[] = {
    {"wait",  (PyCFunction)Barrier_wait,  METH_VARARGS | METH_KEYWORDS, "Wait for all parties"},
    {"reset", (PyCFunction)Barrier_reset, METH_NOARGS,                  "Reset the barrier"},
    {"abort", (PyCFunction)Barrier_abort, METH_NOARGS,                  "Put the barrier into broken state"},
    {NULL}
};

static PyGetSetDef Barrier_getset[] = {
    {"parties",   (getter)Barrier_get_parties,   NULL, "Number of parties required", NULL},
    {"n_waiting", (getter)Barrier_get_n_waiting, NULL, "Number of threads currently waiting", NULL},
    {"broken",    (getter)Barrier_get_broken,    NULL, "Whether the barrier is broken", NULL},
    {NULL}
};

static PyTypeObject BarrierType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._sync.Barrier",
    .tp_doc       = "High-performance C-backed barrier for N-party synchronization.",
    .tp_basicsize = sizeof(BarrierObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = Barrier_new,
    .tp_dealloc   = (destructor)Barrier_dealloc,
    .tp_traverse  = (traverseproc)Barrier_traverse,
    .tp_clear     = (inquiry)Barrier_clear_refs,
    .tp_methods   = Barrier_methods,
    .tp_getset    = Barrier_getset,
};

/* ================================================================
 * MODULE DEFINITION
 * ================================================================ */

static PyObject *
sync_mod_set_enabled(PyObject *Py_UNUSED(self), PyObject *args)
{
    int flag;
    if (!PyArg_ParseTuple(args, "p", &flag))
        return NULL;
    atomic_store_explicit(&cthreading_telemetry_enabled, flag ? 1 : 0, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
sync_mod_enabled(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyMethodDef sync_module_methods[] = {
    {"set_enabled", (PyCFunction)sync_mod_set_enabled, METH_VARARGS, "Enable/disable telemetry for sync primitives"},
    {"enabled",     (PyCFunction)sync_mod_enabled,     METH_NOARGS,  "Query telemetry state"},
    {NULL, NULL, 0, NULL},
};

static PyModuleDef sync_module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name    = "cthreading._sync",
    .m_doc     = "cthreading C core — Lock, RLock, Event, Semaphore, Condition",
    .m_size    = -1,
    .m_methods = sync_module_methods,
};

PyMODINIT_FUNC
PyInit__sync(void)
{
    PyObject *m;

    atomic_init(&cthreading_telemetry_enabled, 0);

    if (PyType_Ready(&LockType) < 0)            return NULL;
    if (PyType_Ready(&RLockType) < 0)           return NULL;
    if (PyType_Ready(&EventType) < 0)           return NULL;
    if (PyType_Ready(&SemaphoreType) < 0)       return NULL;
    if (PyType_Ready(&ConditionType) < 0)       return NULL;
    if (PyType_Ready(&BoundedSemaphoreType) < 0) return NULL;
    if (PyType_Ready(&BarrierType) < 0)         return NULL;

    m = PyModule_Create(&sync_module_def);
    if (m == NULL)
        return NULL;

    /* Create BrokenBarrierError exception */
    BrokenBarrierError = PyErr_NewException(
        "cthreading._sync.BrokenBarrierError", PyExc_RuntimeError, NULL);
    if (!BrokenBarrierError) { Py_DECREF(m); return NULL; }
    Py_INCREF(BrokenBarrierError);
    if (PyModule_AddObject(m, "BrokenBarrierError", BrokenBarrierError) < 0) {
        Py_DECREF(BrokenBarrierError);
        Py_DECREF(m);
        return NULL;
    }

#define ADD_TYPE(name, typeobj)                             \
    Py_INCREF(&(typeobj));                                 \
    if (PyModule_AddObject(m, name, (PyObject *)&(typeobj)) < 0) { \
        Py_DECREF(&(typeobj));                             \
        Py_DECREF(m);                                     \
        return NULL;                                       \
    }

    ADD_TYPE("Lock",              LockType);
    ADD_TYPE("RLock",             RLockType);
    ADD_TYPE("Event",             EventType);
    ADD_TYPE("Semaphore",         SemaphoreType);
    ADD_TYPE("BoundedSemaphore",  BoundedSemaphoreType);
    ADD_TYPE("Condition",         ConditionType);
    ADD_TYPE("Barrier",           BarrierType);

#undef ADD_TYPE

#ifdef Py_MOD_GIL_NOT_USED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    return m;
}
