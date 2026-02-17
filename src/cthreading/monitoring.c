/*
 * src/cthreading/monitoring.c
 *
 * Ghost counters, sharded Counter, telemetry toggle, atomic stats.
 * This module is purely about monitoring / metrics / contention-aware cells.
 * Thread registry and pool management live elsewhere.
 */

#define CTHREADING_MONITORING_IMPL
#include "cthreading_common.h"

/* Global telemetry flag — owned by this TU, extern'd via header */
atomic_int cthreading_telemetry_enabled;

/* ================================================================
 * GHOST OBJECT
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    ct_mutex_t    lock;
    PyObject     *value;
    atomic_llong  int_acc;         /* lock-free integer accumulator        */
    atomic_llong  int_base;        /* lock-free base for int_mode get()    */
    int           int_mode;        /* 1 if value started as int            */
    atomic_ulong  owner;
    atomic_uint   recursion;
    atomic_ullong access_count;
    atomic_ullong contention_count;
    atomic_ullong version;
} GhostObject;

/* --- lifecycle --- */

static int
Ghost_traverse(GhostObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->value);
    return 0;
}

static int
Ghost_clear(GhostObject *self)
{
    Py_CLEAR(self->value);
    return 0;
}

static void
Ghost_dealloc(GhostObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_XDECREF(self->value);
    ct_mutex_destroy(&self->lock);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Ghost_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"initial", NULL};
    PyObject *initial = Py_None;
    GhostObject *self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &initial))
        return NULL;

    self = (GhostObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    ct_mutex_init(&self->lock);

    Py_INCREF(initial);
    self->value = initial;
    atomic_init(&self->int_acc, 0);
    atomic_init(&self->int_base, PyLong_Check(initial) ? PyLong_AsLongLong(initial) : 0);
    self->int_mode = PyLong_Check(initial) ? 1 : 0;
    atomic_init(&self->owner, 0);
    atomic_init(&self->recursion, 0);
    atomic_init(&self->access_count, 0);
    atomic_init(&self->contention_count, 0);
    atomic_init(&self->version, 0);

    return (PyObject *)self;
}

/* --- lock helpers --- */

static void
Ghost_acquire(GhostObject *self)
{
    int contended = !ct_mutex_trylock(&self->lock);
    if (contended) {
        if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
            atomic_fetch_add_explicit(&self->contention_count, 1, memory_order_relaxed);
        ct_mutex_lock(&self->lock);
    }
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed)) {
        atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&self->version, 1, memory_order_relaxed);
    }
}

static void
Ghost_release(GhostObject *self)
{
    ct_mutex_unlock(&self->lock);
}

/* --- Python methods --- */

static PyObject *
Ghost_enter(GhostObject *self, PyObject *Py_UNUSED(ignored))
{
    Ghost_acquire(self);
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
Ghost_exit(GhostObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    Ghost_release(self);
    Py_RETURN_NONE;
}

static PyObject *
Ghost_stats(GhostObject *self, PyObject *Py_UNUSED(ignored))
{
    return Py_BuildValue("{s:K, s:K, s:K}",
        "accesses",  atomic_load_explicit(&self->access_count, memory_order_relaxed),
        "heat",      atomic_load_explicit(&self->contention_count, memory_order_relaxed),
        "version",   atomic_load_explicit(&self->version, memory_order_relaxed));
}

static PyObject *
Ghost_stats_tuple(GhostObject *self, PyObject *Py_UNUSED(ignored))
{
    return Py_BuildValue("(KKK)",
        atomic_load_explicit(&self->access_count, memory_order_relaxed),
        atomic_load_explicit(&self->contention_count, memory_order_relaxed),
        atomic_load_explicit(&self->version, memory_order_relaxed));
}

static PyObject *
Ghost_get(GhostObject *self, PyObject *Py_UNUSED(ignored))
{
    /* Fast path: integer mode — fully lock-free read of base + accumulator */
    if (self->int_mode) {
        long long base_val = atomic_load_explicit(&self->int_base, memory_order_relaxed);
        long long acc = atomic_load_explicit(&self->int_acc, memory_order_relaxed);
        return PyLong_FromLongLong(base_val + acc);
    }
    PyObject *val;
    Ghost_acquire(self);
    val = self->value;
    Py_XINCREF(val);
    Ghost_release(self);
    if (val == NULL)
        Py_RETURN_NONE;
    return val;
}

static PyObject *
Ghost_set(GhostObject *self, PyObject *val)
{
    Ghost_acquire(self);
    /* Reset accumulator — set() replaces the value entirely */
    atomic_store_explicit(&self->int_acc, 0, memory_order_relaxed);
    if (PyLong_Check(val)) {
        self->int_mode = 1;
        atomic_store_explicit(&self->int_base, PyLong_AsLongLong(val), memory_order_relaxed);
    } else {
        self->int_mode = 0;
        atomic_store_explicit(&self->int_base, 0, memory_order_relaxed);
    }
    Py_INCREF(val);
    PyObject *old = self->value;
    self->value = val;
    Py_XDECREF(old);
    Ghost_release(self);
    Py_RETURN_NONE;
}

/* iadd: fast add that returns None (avoids PyLong allocation per call) */
static PyObject *
Ghost_iadd(GhostObject *self, PyObject *delta)
{
    /* FAST PATH: integer delta on integer Ghost — completely lock-free */
    if (self->int_mode && PyLong_Check(delta)) {
        int overflow;
        long long d = PyLong_AsLongLongAndOverflow(delta, &overflow);
        if (d == -1 && PyErr_Occurred())
            return NULL;
        if (!overflow) {
            atomic_fetch_add_explicit(&self->int_acc, d, memory_order_relaxed);
            if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed)) {
                atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
                atomic_fetch_add_explicit(&self->version, 1, memory_order_relaxed);
            }
            Py_RETURN_NONE;
        }
    }

    /* SLOW PATH: general PyObject add under lock */
    Ghost_acquire(self);
    PyObject *current = self->value;
    if (current == NULL || current == Py_None) {
        current = PyLong_FromLong(0);
    } else {
        Py_INCREF(current);
    }
    PyObject *result = PyNumber_Add(current, delta);
    Py_DECREF(current);
    if (result == NULL) {
        Ghost_release(self);
        return NULL;
    }
    PyObject *old = self->value;
    Py_INCREF(result);
    self->value = result;
    Py_XDECREF(old);
    Ghost_release(self);
    Py_DECREF(result);
    Py_RETURN_NONE;
}

static PyObject *
Ghost_add(GhostObject *self, PyObject *delta)
{
    /* FAST PATH: integer delta on integer Ghost — completely lock-free */
    if (self->int_mode && PyLong_Check(delta)) {
        int overflow;
        long long d = PyLong_AsLongLongAndOverflow(delta, &overflow);
        if (d == -1 && PyErr_Occurred())
            return NULL;
        if (!overflow) {
            long long new_val = atomic_fetch_add_explicit(
                &self->int_acc, d, memory_order_relaxed) + d;
            if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed)) {
                atomic_fetch_add_explicit(&self->access_count, 1, memory_order_relaxed);
                atomic_fetch_add_explicit(&self->version, 1, memory_order_relaxed);
            }
            long long base_val = atomic_load_explicit(&self->int_base, memory_order_relaxed);
            return PyLong_FromLongLong(base_val + new_val);
        }
    }

    /* SLOW PATH: general PyObject add under lock */
    Ghost_acquire(self);
    PyObject *current = self->value;
    if (current == NULL || current == Py_None) {
        current = PyLong_FromLong(0);
    } else {
        Py_INCREF(current);
    }
    PyObject *result = PyNumber_Add(current, delta);
    Py_DECREF(current);
    if (result == NULL) {
        Ghost_release(self);
        return NULL;
    }
    PyObject *old = self->value;
    Py_INCREF(result);
    self->value = result;
    Py_XDECREF(old);
    Ghost_release(self);
    return result;
}

static PyObject *
Ghost_update(GhostObject *self, PyObject *fn)
{
    Ghost_acquire(self);
    PyObject *current = self->value;
    Py_XINCREF(current);
    if (current == NULL) {
        current = Py_None;
        Py_INCREF(current);
    }
    PyObject *result = PyObject_CallOneArg(fn, current);
    Py_DECREF(current);
    if (result == NULL) {
        Ghost_release(self);
        return NULL;
    }
    PyObject *old = self->value;
    Py_INCREF(result);
    self->value = result;
    Py_XDECREF(old);
    Ghost_release(self);
    return result;
}

static PyMethodDef Ghost_methods[] = {
    {"__enter__",   (PyCFunction)Ghost_enter,       METH_NOARGS,   "Enter the lock"},
    {"__exit__",    (PyCFunction)Ghost_exit,         METH_FASTCALL, "Exit the lock"},
    {"stats",       (PyCFunction)Ghost_stats,        METH_NOARGS,   "Get contention telemetry"},
    {"stats_tuple", (PyCFunction)Ghost_stats_tuple,  METH_NOARGS,   "Get contention telemetry as a tuple"},
    {"get",         (PyCFunction)Ghost_get,          METH_NOARGS,   "Get the stored value"},
    {"set",         (PyCFunction)Ghost_set,          METH_O,        "Set the stored value"},
    {"add",         (PyCFunction)Ghost_add,          METH_O,        "Add delta (numeric types)"},
    {"iadd",        (PyCFunction)Ghost_iadd,         METH_O,        "Add delta in-place, returns None (fast)"},
    {"update",      (PyCFunction)Ghost_update,       METH_O,        "Atomic read-modify-write via callable"},
    {NULL}
};

static PyTypeObject GhostType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._monitoring.Ghost",
    .tp_doc       = "A re-entrant locked cell for any Python object with contention telemetry.",
    .tp_basicsize = sizeof(GhostObject),
    .tp_itemsize  = 0,
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = Ghost_new,
    .tp_dealloc   = (destructor)Ghost_dealloc,
    .tp_traverse  = (traverseproc)Ghost_traverse,
    .tp_clear     = (inquiry)Ghost_clear,
    .tp_methods   = Ghost_methods,
};

/* ================================================================
 * COUNTER OBJECT (sharded int64)
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    Py_ssize_t    shard_count;
    CounterShard *shards;
} CounterObject;

static void
Counter_dealloc(CounterObject *self)
{
    if (self->shards)
        PyMem_Free(self->shards);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Counter_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    CounterObject *self;
    static char *kwlist[] = {"initial", "shards", NULL};
    long long initial = 0;
    Py_ssize_t shards = 64;

    self = (CounterObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|Ln", kwlist, &initial, &shards)) {
        Py_DECREF(self);
        return NULL;
    }
    if (shards <= 0) {
        PyErr_SetString(PyExc_ValueError, "shards must be > 0");
        Py_DECREF(self);
        return NULL;
    }

    self->shard_count = shards;
    self->shards = (CounterShard *)PyMem_Calloc((size_t)shards, sizeof(CounterShard));
    if (self->shards == NULL) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < shards; i++) {
        atomic_init(&self->shards[i].access_count, 0);
        atomic_init(&self->shards[i].contention_count, 0);
        atomic_init(&self->shards[i].version, 0);
        atomic_init(&self->shards[i].value, 0);
    }
    atomic_store_explicit(&self->shards[0].value, initial, memory_order_relaxed);
    return (PyObject *)self;
}

static inline Py_ssize_t
Counter_pick_shard(CounterObject *self)
{
    unsigned long tid = PyThread_get_thread_ident();
    return (Py_ssize_t)(tid % (unsigned long)self->shard_count);
}

static inline void
Counter_shard_telemetry(CounterShard *shard)
{
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed)) {
        atomic_fetch_add_explicit(&shard->access_count, 1, memory_order_relaxed);
        atomic_fetch_add_explicit(&shard->version, 1, memory_order_relaxed);
    }
}

static PyObject *
Counter_enter(CounterObject *self, PyObject *Py_UNUSED(ignored))
{
    /* Lock-free counter: enter/exit are no-ops for compatibility */
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *
Counter_exit(CounterObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    Py_RETURN_NONE;
}

static PyObject *
Counter_add(CounterObject *self, PyObject *delta_obj)
{
    long long delta = PyLong_AsLongLong(delta_obj);
    if (delta == -1 && PyErr_Occurred())
        return NULL;
    Py_ssize_t idx = Counter_pick_shard(self);
    CounterShard *shard = &self->shards[idx];
    atomic_fetch_add_explicit(&shard->value, delta, memory_order_relaxed);
    Counter_shard_telemetry(shard);
    Py_RETURN_NONE;
}

static PyObject *
Counter_get(CounterObject *self, PyObject *Py_UNUSED(ignored))
{
    long long total = 0;
    for (Py_ssize_t i = 0; i < self->shard_count; i++)
        total += atomic_load_explicit(&self->shards[i].value, memory_order_relaxed);
    return PyLong_FromLongLong(total);
}

static PyObject *
Counter_set(CounterObject *self, PyObject *args)
{
    long long value;
    if (!PyArg_ParseTuple(args, "L", &value))
        return NULL;
    /* Reset all shards to 0, set shard 0 to the new value */
    for (Py_ssize_t i = 0; i < self->shard_count; i++)
        atomic_store_explicit(&self->shards[i].value, 0, memory_order_relaxed);
    atomic_store_explicit(&self->shards[0].value, value, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
Counter_stats_tuple(CounterObject *self, PyObject *Py_UNUSED(ignored))
{
    unsigned long long a = 0, h = 0, v = 0;
    for (Py_ssize_t i = 0; i < self->shard_count; i++) {
        a += atomic_load_explicit(&self->shards[i].access_count, memory_order_relaxed);
        h += atomic_load_explicit(&self->shards[i].contention_count, memory_order_relaxed);
        v += atomic_load_explicit(&self->shards[i].version, memory_order_relaxed);
    }
    return Py_BuildValue("(KKK)", a, h, v);
}

static PyObject *
Counter_stats(CounterObject *self, PyObject *Py_UNUSED(ignored))
{
    PyObject *tup = Counter_stats_tuple(self, NULL);
    if (tup == NULL)
        return NULL;
    unsigned long long a, h, v;
    if (!PyArg_ParseTuple(tup, "KKK", &a, &h, &v)) {
        Py_DECREF(tup);
        return NULL;
    }
    Py_DECREF(tup);
    return Py_BuildValue("{s:K, s:K, s:K}", "accesses", a, "heat", h, "version", v);
}

static PyMethodDef Counter_methods[] = {
    {"__enter__",   (PyCFunction)Counter_enter,       METH_NOARGS,   "Enter exclusive region over all shards"},
    {"__exit__",    (PyCFunction)Counter_exit,         METH_FASTCALL, "Exit exclusive region over all shards"},
    {"add",         (PyCFunction)Counter_add,          METH_O,        "Add delta to the sharded counter"},
    {"get",         (PyCFunction)Counter_get,          METH_NOARGS,   "Get the global counter value"},
    {"set",         (PyCFunction)Counter_set,          METH_VARARGS,  "Set the global counter value"},
    {"stats",       (PyCFunction)Counter_stats,        METH_NOARGS,   "Get contention telemetry"},
    {"stats_tuple", (PyCFunction)Counter_stats_tuple,  METH_NOARGS,   "Get contention telemetry as a tuple"},
    {NULL}
};

static PyTypeObject CounterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._monitoring.Counter",
    .tp_doc       = "A sharded contention-aware int counter.",
    .tp_basicsize = sizeof(CounterObject),
    .tp_itemsize  = 0,
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new       = Counter_new,
    .tp_dealloc   = (destructor)Counter_dealloc,
    .tp_methods   = Counter_methods,
};

/* ================================================================
 * MODULE-LEVEL: telemetry toggle
 * ================================================================ */

static PyObject *
mod_set_enabled(PyObject *Py_UNUSED(self), PyObject *args)
{
    int flag;
    if (!PyArg_ParseTuple(args, "p", &flag))
        return NULL;
    atomic_store_explicit(&cthreading_telemetry_enabled, flag ? 1 : 0, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
mod_enabled(PyObject *Py_UNUSED(self), PyObject *Py_UNUSED(args))
{
    if (atomic_load_explicit(&cthreading_telemetry_enabled, memory_order_relaxed))
        Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

/* ================================================================
 * MODULE DEFINITION
 * ================================================================ */

static PyMethodDef module_methods[] = {
    {"set_enabled", (PyCFunction)mod_set_enabled, METH_VARARGS, "Enable/disable telemetry"},
    {"enabled",     (PyCFunction)mod_enabled,     METH_NOARGS,  "Query telemetry state"},
    {NULL, NULL, 0, NULL},
};

static PyModuleDef monitoring_module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name    = "cthreading._monitoring",
    .m_doc     = "cthreading C core — Ghost cells, sharded counters, telemetry",
    .m_size    = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC
PyInit__monitoring(void)
{
    PyObject *m;

    atomic_init(&cthreading_telemetry_enabled, 0);

    if (PyType_Ready(&GhostType) < 0)
        return NULL;
    if (PyType_Ready(&CounterType) < 0)
        return NULL;

    m = PyModule_Create(&monitoring_module_def);
    if (m == NULL)
        return NULL;

    Py_INCREF(&GhostType);
    if (PyModule_AddObject(m, "Ghost", (PyObject *)&GhostType) < 0) {
        Py_DECREF(&GhostType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&CounterType);
    if (PyModule_AddObject(m, "Counter", (PyObject *)&CounterType) < 0) {
        Py_DECREF(&CounterType);
        Py_DECREF(m);
        return NULL;
    }

#ifdef Py_MOD_GIL_NOT_USED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    return m;
}
