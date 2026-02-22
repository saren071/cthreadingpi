/*
 * src/cthreading/tasks.c
 *
 * Task batching, flush support, and task group management.
 *   - TaskBatch    : accumulates tasks and flushes to a pool
 *   - TaskGroup    : groups related tasks with a shared ID for tracking
 *
 * Integrates with monitoring for counters/metrics.
 */

#include "cthreading_common.h"

/* Forward declare ThreadPool submit helper */
static int
batch_submit_node(PyObject *pool_obj, TaskNode *node);

/* ================================================================
 * TASKBATCH OBJECT
 * ================================================================ */

/* BatchObject is defined in cthreading_common.h */

static int
TaskBatch_traverse(BatchObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->pool);
    TaskNode *n = self->head;
    while (n) {
        Py_VISIT(n->callable);
        Py_VISIT(n->args);
        Py_VISIT(n->kwargs);
        n = n->next;
    }
    return 0;
}

static int
TaskBatch_clear_tasks(BatchObject *self)
{
    TaskNode *n = self->head;
    while (n) {
        TaskNode *next = n->next;
        Py_CLEAR(n->callable);
        Py_CLEAR(n->args);
        Py_XDECREF(n->kwargs);
        n->kwargs = NULL;
        PyMem_Free(n);
        n = next;
    }
    self->head = self->tail = NULL;
    self->count = 0;
    return 0;
}

static void
TaskBatch_dealloc(BatchObject *self)
{
    PyObject_GC_UnTrack(self);
    TaskBatch_clear_tasks(self);
    Py_XDECREF(self->pool);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
TaskBatch_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"pool", "flush_threshold", "priority", "group", NULL};
    PyObject *pool_obj;
    Py_ssize_t flush_threshold = 100;
    int priority = 0;
    long long group = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|niL", kwlist,
                                     &pool_obj, &flush_threshold, &priority, &group))
        return NULL;

    BatchObject *self = (BatchObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    Py_INCREF(pool_obj);
    self->pool = pool_obj;
    self->head = self->tail = NULL;
    self->count = 0;
    self->flush_threshold = flush_threshold;
    self->default_priority = priority;
    self->default_group = group;

    return (PyObject *)self;
}

static PyObject *
TaskBatch_add(BatchObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "args", "kwargs", "priority", NULL};
    PyObject *fn;
    PyObject *fn_args = NULL;
    PyObject *fn_kwargs = NULL;
    int priority = -999;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOi", kwlist,
                                     &fn, &fn_args, &fn_kwargs, &priority))
        return NULL;

    if (!PyCallable_Check(fn)) {
        PyErr_SetString(PyExc_TypeError, "fn must be callable");
        return NULL;
    }

    if (priority == -999)
        priority = self->default_priority;

    TaskNode *node = (TaskNode *)PyMem_Calloc(1, sizeof(TaskNode));
    if (node == NULL)
        return PyErr_NoMemory();

    Py_INCREF(fn);
    node->callable = fn;

    if (fn_args == NULL || fn_args == Py_None) {
        node->args = PyTuple_New(0);
    } else if (PyTuple_Check(fn_args)) {
        Py_INCREF(fn_args);
        node->args = fn_args;
    } else {
        node->args = PySequence_Tuple(fn_args);
        if (node->args == NULL) {
            Py_DECREF(fn);
            PyMem_Free(node);
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
    node->group_id = self->default_group;
    node->next = NULL;

    /* Append to batch list */
    if (self->tail) {
        self->tail->next = node;
    } else {
        self->head = node;
    }
    self->tail = node;
    self->count++;

    /* Auto-flush if threshold reached */
    if (self->flush_threshold > 0 && self->count >= self->flush_threshold) {
        /* Flush inline */
        TaskNode *n = self->head;
        while (n) {
            TaskNode *next = n->next;
            if (batch_submit_node(self->pool, n) < 0) {
                /* On error, keep remaining in batch */
                self->head = n;
                /* Recount */
                Py_ssize_t remaining = 0;
                TaskNode *r = n;
                while (r) { remaining++; r = r->next; }
                self->count = remaining;
                return NULL;
            }
            n = next;
        }
        self->head = self->tail = NULL;
        self->count = 0;
    }

    Py_RETURN_NONE;
}

static PyObject *
TaskBatch_flush(BatchObject *self, PyObject *Py_UNUSED(ignored))
{
    TaskNode *n = self->head;
    Py_ssize_t flushed = 0;
    while (n) {
        TaskNode *next = n->next;
        if (batch_submit_node(self->pool, n) < 0) {
            self->head = n;
            Py_ssize_t remaining = 0;
            TaskNode *r = n;
            while (r) { remaining++; r = r->next; }
            self->count = remaining;
            return NULL;
        }
        flushed++;
        n = next;
    }
    self->head = self->tail = NULL;
    self->count = 0;
    return PyLong_FromSsize_t(flushed);
}

static PyObject *
TaskBatch_pending(BatchObject *self, PyObject *Py_UNUSED(ignored))
{
    return PyLong_FromSsize_t(self->count);
}

static PyMethodDef TaskBatch_methods[] = {
    {"add",     (PyCFunction)TaskBatch_add,     METH_VARARGS | METH_KEYWORDS, "Add a task to the batch"},
    {"flush",   (PyCFunction)TaskBatch_flush,   METH_NOARGS,                  "Flush all pending tasks to the pool"},
    {"pending", (PyCFunction)TaskBatch_pending,  METH_NOARGS,                  "Number of pending tasks"},
    {NULL}
};

static PyTypeObject TaskBatchType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._tasks.TaskBatch",
    .tp_doc       = "Accumulates tasks and flushes them to a ThreadPool in batch.",
    .tp_basicsize = sizeof(BatchObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = TaskBatch_new,
    .tp_dealloc   = (destructor)TaskBatch_dealloc,
    .tp_traverse  = (traverseproc)TaskBatch_traverse,
    .tp_clear     = (inquiry)TaskBatch_clear_tasks,
    .tp_methods   = TaskBatch_methods,
};

/* ================================================================
 * TASKGROUP OBJECT
 * ================================================================ */

typedef struct {
    PyObject_HEAD
    int64_t          group_id;
    PyObject        *pool;          /* back-reference to ThreadPool */
    atomic_llong     submitted;
    atomic_llong     completed;
    atomic_llong     failed;
} TaskGroupObject;

static int64_t _next_group_id = 1;

static int
TaskGroup_traverse(TaskGroupObject *self, visitproc visit, void *arg)
{
    Py_VISIT(self->pool);
    return 0;
}

static int
TaskGroup_clear_refs(TaskGroupObject *self)
{
    Py_CLEAR(self->pool);
    return 0;
}

static void
TaskGroup_dealloc(TaskGroupObject *self)
{
    PyObject_GC_UnTrack(self);
    Py_XDECREF(self->pool);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
TaskGroup_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"pool", NULL};
    PyObject *pool_obj;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &pool_obj))
        return NULL;

    TaskGroupObject *self = (TaskGroupObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    Py_INCREF(pool_obj);
    self->pool = pool_obj;
    self->group_id = _next_group_id++;
    atomic_init(&self->submitted, 0);
    atomic_init(&self->completed, 0);
    atomic_init(&self->failed, 0);

    return (PyObject *)self;
}

static PyObject *
TaskGroup_submit(TaskGroupObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"fn", "args", "kwargs", "priority", NULL};
    PyObject *fn;
    PyObject *fn_args = NULL;
    PyObject *fn_kwargs = NULL;
    int priority = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOi", kwlist,
                                     &fn, &fn_args, &fn_kwargs, &priority))
        return NULL;

    /* Build submit call with group ID */
    PyObject *call_args = Py_BuildValue("(OOOLL)",
        fn,
        fn_args ? fn_args : Py_None,
        fn_kwargs ? fn_kwargs : Py_None,
        priority,
        (long long)self->group_id);
    if (call_args == NULL)
        return NULL;

    PyObject *r = PyObject_CallMethod(self->pool, "submit", "OOOiL",
        fn,
        fn_args ? fn_args : Py_None,
        fn_kwargs ? fn_kwargs : Py_None,
        priority,
        (long long)self->group_id);
    Py_DECREF(call_args);

    if (r == NULL)
        return NULL;
    Py_DECREF(r);

    atomic_fetch_add_explicit(&self->submitted, 1, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
TaskGroup_stats(TaskGroupObject *self, PyObject *Py_UNUSED(ignored))
{
    return Py_BuildValue("{s:L, s:L, s:L, s:L}",
        "group_id",  (long long)self->group_id,
        "submitted", atomic_load_explicit(&self->submitted, memory_order_relaxed),
        "completed", atomic_load_explicit(&self->completed, memory_order_relaxed),
        "failed",    atomic_load_explicit(&self->failed, memory_order_relaxed));
}

static PyMethodDef TaskGroup_methods[] = {
    {"submit", (PyCFunction)TaskGroup_submit, METH_VARARGS | METH_KEYWORDS, "Submit a task in this group"},
    {"stats",  (PyCFunction)TaskGroup_stats,  METH_NOARGS,                  "Get group statistics"},
    {NULL}
};

static PyTypeObject TaskGroupType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._tasks.TaskGroup",
    .tp_doc       = "Groups related tasks for tracking.",
    .tp_basicsize = sizeof(TaskGroupObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = TaskGroup_new,
    .tp_dealloc   = (destructor)TaskGroup_dealloc,
    .tp_traverse  = (traverseproc)TaskGroup_traverse,
    .tp_clear     = (inquiry)TaskGroup_clear_refs,
    .tp_methods   = TaskGroup_methods,
};

/* ================================================================
 * HELPER: submit a TaskNode to a pool object
 * ================================================================ */

static int
batch_submit_node(PyObject *pool_obj, TaskNode *node)
{
    /* Call pool.submit(fn, args, kwargs, priority, group) */
    PyObject *r = PyObject_CallMethod(pool_obj, "submit", "OOOiL",
        node->callable,
        node->args,
        node->kwargs ? node->kwargs : Py_None,
        node->priority,
        (long long)node->group_id);

    /* Free the node resources (pool's submit makes copies) */
    Py_DECREF(node->callable);
    Py_DECREF(node->args);
    Py_XDECREF(node->kwargs);
    PyMem_Free(node);

    if (r == NULL)
        return -1;
    Py_DECREF(r);
    return 0;
}

/* ================================================================
 * MODULE DEFINITION
 * ================================================================ */

static PyMethodDef tasks_module_methods[] = {
    {NULL, NULL, 0, NULL},
};

static PyModuleDef tasks_module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name    = "cthreading._tasks",
    .m_doc     = "cthreading C core — TaskBatch and TaskGroup",
    .m_size    = -1,
    .m_methods = tasks_module_methods,
};

PyMODINIT_FUNC
PyInit__tasks(void)
{
    PyObject *m;

    if (PyType_Ready(&TaskBatchType) < 0) return NULL;
    if (PyType_Ready(&TaskGroupType) < 0) return NULL;

    m = PyModule_Create(&tasks_module_def);
    if (m == NULL)
        return NULL;

    Py_INCREF(&TaskBatchType);
    if (PyModule_AddObject(m, "TaskBatch", (PyObject *)&TaskBatchType) < 0) {
        Py_DECREF(&TaskBatchType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&TaskGroupType);
    if (PyModule_AddObject(m, "TaskGroup", (PyObject *)&TaskGroupType) < 0) {
        Py_DECREF(&TaskGroupType);
        Py_DECREF(m);
        return NULL;
    }

#ifdef Py_MOD_GIL_NOT_USED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    return m;
}
