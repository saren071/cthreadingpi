/*
 * src/cthreading/queue.c
 *
 * Thread-safe queues implemented in C:
 *   - Queue          (FIFO, bounded or unbounded)
 *   - PriorityQueue  (min-heap, bounded or unbounded)
 *   - LifoQueue      (LIFO/stack, bounded or unbounded)
 *   - SimpleQueue    (unbounded FIFO, no task tracking)
 *
 * All support blocking put/get with proper timeout.
 * Raises Empty/Full exceptions matching stdlib queue module.
 */

#include "cthreading_common.h"

/* Exception objects — initialized in module init */
static PyObject *QueueEmpty;
static PyObject *QueueFull;

/* ================================================================
 * QUEUE (FIFO)
 * ================================================================ */

typedef struct QueueNode {
    PyObject          *item;
    struct QueueNode  *next;
} QueueNode;

typedef struct {
    PyObject_HEAD
    ct_mutex_t         mutex;
    ct_cond_t          not_empty;
    ct_cond_t          not_full;
    ct_cond_t          all_done;     /* for join() */
    QueueNode         *head;
    QueueNode         *tail;
    QueueNode         *free_list;   /* recycled nodes    */
    Py_ssize_t         free_count;
    Py_ssize_t         size;
    Py_ssize_t         maxsize;     /* 0 = unbounded    */
    Py_ssize_t         unfinished;  /* for task_done/join */
    atomic_llong       total_put;
    atomic_llong       total_get;
} QueueObject;

static int
Queue_traverse(QueueObject *self, visitproc visit, void *arg)
{
    QueueNode *n = self->head;
    while (n) {
        Py_VISIT(n->item);
        n = n->next;
    }
    return 0;
}

static int
Queue_clear_items(QueueObject *self)
{
    QueueNode *n = self->head;
    while (n) {
        QueueNode *next = n->next;
        Py_CLEAR(n->item);
        PyMem_Free(n);
        n = next;
    }
    self->head = self->tail = NULL;
    self->size = 0;
    return 0;
}

static void
Queue_dealloc(QueueObject *self)
{
    PyObject_GC_UnTrack(self);
    Queue_clear_items(self);
    /* Free the free-list */
    QueueNode *fn = self->free_list;
    while (fn) {
        QueueNode *next = fn->next;
        PyMem_Free(fn);
        fn = next;
    }
    self->free_list = NULL;
    self->free_count = 0;
    ct_cond_destroy(&self->all_done);
    ct_cond_destroy(&self->not_full);
    ct_cond_destroy(&self->not_empty);
    ct_mutex_destroy(&self->mutex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Queue_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"maxsize", NULL};
    Py_ssize_t maxsize = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &maxsize))
        return NULL;

    QueueObject *self = (QueueObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    ct_mutex_init(&self->mutex);
    ct_cond_init(&self->not_empty);
    ct_cond_init(&self->not_full);
    ct_cond_init(&self->all_done);

    self->head = self->tail = NULL;
    self->free_list = NULL;
    self->free_count = 0;
    self->size = 0;
    self->maxsize = maxsize;
    self->unfinished = 0;
    atomic_init(&self->total_put, 0);
    atomic_init(&self->total_get, 0);

    return (PyObject *)self;
}

static PyObject *
Queue_put(QueueObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"item", "blocking", "timeout", NULL};
    PyObject *item;
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|pd", kwlist, &item, &blocking, &timeout))
        return NULL;

    Py_INCREF(item);

    ct_mutex_lock(&self->mutex);

    /* Wait for space if bounded */
    if (self->maxsize > 0 && self->size >= self->maxsize) {
        if (!blocking) {
            ct_mutex_unlock(&self->mutex);
            Py_DECREF(item);
            PyErr_SetNone(QueueFull);
            return NULL;
        }
        if (timeout < 0) {
            while (self->size >= self->maxsize)
                ct_cond_wait(&self->not_full, &self->mutex);
        } else {
            double deadline = ct_time_ms() + timeout * 1000.0;
            while (self->size >= self->maxsize) {
                double remaining = deadline - ct_time_ms();
                if (remaining <= 0) {
                    ct_mutex_unlock(&self->mutex);
                    Py_DECREF(item);
                    PyErr_SetNone(QueueFull);
                    return NULL;
                }
                unsigned long ms = (unsigned long)remaining;
                if (ms == 0) ms = 1;
                ct_cond_timedwait_ms(&self->not_full, &self->mutex, ms);
            }
        }
    }

    /* Try to reuse a node from the free-list */
    QueueNode *node = self->free_list;
    if (node) {
        self->free_list = node->next;
        self->free_count--;
    } else {
        node = (QueueNode *)PyMem_Calloc(1, sizeof(QueueNode));
        if (node == NULL) {
            ct_mutex_unlock(&self->mutex);
            Py_DECREF(item);
            return PyErr_NoMemory();
        }
    }
    node->item = item;
    node->next = NULL;

    if (self->tail)
        self->tail->next = node;
    else
        self->head = node;
    self->tail = node;
    self->size++;
    self->unfinished++;

    ct_cond_signal(&self->not_empty);
    ct_mutex_unlock(&self->mutex);

    atomic_fetch_add_explicit(&self->total_put, 1, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
Queue_put_nowait(QueueObject *self, PyObject *args)
{
    PyObject *item;
    if (!PyArg_ParseTuple(args, "O", &item))
        return NULL;

    Py_INCREF(item);
    ct_mutex_lock(&self->mutex);

    if (self->maxsize > 0 && self->size >= self->maxsize) {
        ct_mutex_unlock(&self->mutex);
        Py_DECREF(item);
        PyErr_SetNone(QueueFull);
        return NULL;
    }

    QueueNode *node = self->free_list;
    if (node) {
        self->free_list = node->next;
        self->free_count--;
    } else {
        node = (QueueNode *)PyMem_Calloc(1, sizeof(QueueNode));
        if (node == NULL) {
            ct_mutex_unlock(&self->mutex);
            Py_DECREF(item);
            return PyErr_NoMemory();
        }
    }
    node->item = item;
    node->next = NULL;

    if (self->tail)
        self->tail->next = node;
    else
        self->head = node;
    self->tail = node;
    self->size++;
    self->unfinished++;

    ct_cond_signal(&self->not_empty);
    ct_mutex_unlock(&self->mutex);

    atomic_fetch_add_explicit(&self->total_put, 1, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
Queue_get(QueueObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"blocking", "timeout", NULL};
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pd", kwlist, &blocking, &timeout))
        return NULL;

    ct_mutex_lock(&self->mutex);

    if (self->size == 0) {
        if (!blocking) {
            ct_mutex_unlock(&self->mutex);
            PyErr_SetNone(QueueEmpty);
            return NULL;
        }
        if (timeout < 0) {
            while (self->size == 0)
                ct_cond_wait(&self->not_empty, &self->mutex);
        } else {
            double deadline = ct_time_ms() + timeout * 1000.0;
            while (self->size == 0) {
                double remaining = deadline - ct_time_ms();
                if (remaining <= 0) {
                    ct_mutex_unlock(&self->mutex);
                    PyErr_SetNone(QueueEmpty);
                    return NULL;
                }
                unsigned long ms = (unsigned long)remaining;
                if (ms == 0) ms = 1;
                ct_cond_timedwait_ms(&self->not_empty, &self->mutex, ms);
            }
        }
    }

    QueueNode *node = self->head;
    self->head = node->next;
    if (self->head == NULL)
        self->tail = NULL;
    self->size--;

    if (self->maxsize > 0)
        ct_cond_signal(&self->not_full);

    ct_mutex_unlock(&self->mutex);

    PyObject *item = node->item;
    /* Return node to free-list (max 1024 cached nodes) */
    node->item = NULL;
    ct_mutex_lock(&self->mutex);
    if (self->free_count < 1024) {
        node->next = self->free_list;
        self->free_list = node;
        self->free_count++;
        ct_mutex_unlock(&self->mutex);
    } else {
        ct_mutex_unlock(&self->mutex);
        PyMem_Free(node);
    }
    atomic_fetch_add_explicit(&self->total_get, 1, memory_order_relaxed);
    return item;
}

static PyObject *
Queue_get_nowait(QueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);

    if (self->size == 0) {
        ct_mutex_unlock(&self->mutex);
        PyErr_SetNone(QueueEmpty);
        return NULL;
    }

    QueueNode *node = self->head;
    self->head = node->next;
    if (self->head == NULL)
        self->tail = NULL;
    self->size--;

    if (self->maxsize > 0)
        ct_cond_signal(&self->not_full);

    ct_mutex_unlock(&self->mutex);

    PyObject *item = node->item;
    node->item = NULL;
    ct_mutex_lock(&self->mutex);
    if (self->free_count < 1024) {
        node->next = self->free_list;
        self->free_list = node;
        self->free_count++;
        ct_mutex_unlock(&self->mutex);
    } else {
        ct_mutex_unlock(&self->mutex);
        PyMem_Free(node);
    }
    atomic_fetch_add_explicit(&self->total_get, 1, memory_order_relaxed);
    return item;
}

static PyObject *
Queue_task_done(QueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    if (self->unfinished <= 0) {
        ct_mutex_unlock(&self->mutex);
        PyErr_SetString(PyExc_ValueError, "task_done() called too many times");
        return NULL;
    }
    self->unfinished--;
    if (self->unfinished == 0)
        ct_cond_broadcast(&self->all_done);
    ct_mutex_unlock(&self->mutex);
    Py_RETURN_NONE;
}

static PyObject *
Queue_join(QueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    while (self->unfinished > 0)
        ct_cond_wait(&self->all_done, &self->mutex);
    ct_mutex_unlock(&self->mutex);
    Py_RETURN_NONE;
}

static PyObject *
Queue_qsize(QueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    Py_ssize_t s = self->size;
    ct_mutex_unlock(&self->mutex);
    return PyLong_FromSsize_t(s);
}

static PyObject *
Queue_empty(QueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    int e = (self->size == 0);
    ct_mutex_unlock(&self->mutex);
    if (e) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Queue_full(QueueObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->maxsize <= 0)
        Py_RETURN_FALSE;
    ct_mutex_lock(&self->mutex);
    int f = (self->size >= self->maxsize);
    ct_mutex_unlock(&self->mutex);
    if (f) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Queue_stats(QueueObject *self, PyObject *Py_UNUSED(ignored))
{
    return Py_BuildValue("{s:L, s:L, s:n}",
        "total_put", atomic_load_explicit(&self->total_put, memory_order_relaxed),
        "total_get", atomic_load_explicit(&self->total_get, memory_order_relaxed),
        "current_size", self->size);
}

static PyMethodDef Queue_methods[] = {
    {"put",         (PyCFunction)Queue_put,         METH_VARARGS | METH_KEYWORDS, "Put an item into the queue"},
    {"put_nowait",  (PyCFunction)Queue_put_nowait,  METH_VARARGS,                 "Put an item without blocking"},
    {"get",         (PyCFunction)Queue_get,         METH_VARARGS | METH_KEYWORDS, "Get an item from the queue"},
    {"get_nowait",  (PyCFunction)Queue_get_nowait,  METH_NOARGS,                  "Get an item without blocking"},
    {"task_done",   (PyCFunction)Queue_task_done,   METH_NOARGS,                  "Signal that a task is complete"},
    {"join",        (PyCFunction)Queue_join,         METH_NOARGS,                  "Block until all tasks are done"},
    {"qsize",       (PyCFunction)Queue_qsize,       METH_NOARGS,                  "Return the queue size"},
    {"empty",       (PyCFunction)Queue_empty,       METH_NOARGS,                  "Return True if empty"},
    {"full",        (PyCFunction)Queue_full,        METH_NOARGS,                  "Return True if full"},
    {"stats",       (PyCFunction)Queue_stats,       METH_NOARGS,                  "Get queue statistics"},
    {NULL}
};

static PyTypeObject QueueType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._queue.Queue",
    .tp_doc       = "High-performance thread-safe FIFO queue.",
    .tp_basicsize = sizeof(QueueObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = Queue_new,
    .tp_dealloc   = (destructor)Queue_dealloc,
    .tp_traverse  = (traverseproc)Queue_traverse,
    .tp_clear     = (inquiry)Queue_clear_items,
    .tp_methods   = Queue_methods,
};

/* ================================================================
 * PRIORITY QUEUE (min-heap)
 * ================================================================ */

typedef struct {
    PyObject *priority;   /* comparison key (strong ref) */
    PyObject *item;       /* the actual item (strong ref) */
} PQEntry;

typedef struct {
    PyObject_HEAD
    ct_mutex_t         mutex;
    ct_cond_t          not_empty;
    PQEntry           *heap;
    Py_ssize_t         size;
    Py_ssize_t         capacity;
    Py_ssize_t         maxsize;    /* 0 = unbounded */
    atomic_llong       total_put;
    atomic_llong       total_get;
} PriorityQueueObject;

static int
PQ_traverse(PriorityQueueObject *self, visitproc visit, void *arg)
{
    for (Py_ssize_t i = 0; i < self->size; i++) {
        Py_VISIT(self->heap[i].priority);
        Py_VISIT(self->heap[i].item);
    }
    return 0;
}

static int
PQ_clear_items(PriorityQueueObject *self)
{
    for (Py_ssize_t i = 0; i < self->size; i++) {
        Py_CLEAR(self->heap[i].priority);
        Py_CLEAR(self->heap[i].item);
    }
    self->size = 0;
    return 0;
}

static void
PQ_dealloc(PriorityQueueObject *self)
{
    PyObject_GC_UnTrack(self);
    PQ_clear_items(self);
    if (self->heap)
        PyMem_Free(self->heap);
    ct_cond_destroy(&self->not_empty);
    ct_mutex_destroy(&self->mutex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
PQ_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"maxsize", NULL};
    Py_ssize_t maxsize = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &maxsize))
        return NULL;

    PriorityQueueObject *self = (PriorityQueueObject *)type->tp_alloc(type, 0);
    if (self == NULL)
        return NULL;

    ct_mutex_init(&self->mutex);
    ct_cond_init(&self->not_empty);

    Py_ssize_t init_cap = 64;
    self->heap = (PQEntry *)PyMem_Calloc((size_t)init_cap, sizeof(PQEntry));
    if (self->heap == NULL) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    self->capacity = init_cap;
    self->size = 0;
    self->maxsize = maxsize;
    atomic_init(&self->total_put, 0);
    atomic_init(&self->total_get, 0);

    return (PyObject *)self;
}

static inline void
PQ_swap(PQEntry *a, PQEntry *b)
{
    PQEntry tmp = *a;
    *a = *b;
    *b = tmp;
}

static void
PQ_sift_up(PQEntry *heap, Py_ssize_t pos)
{
    while (pos > 0) {
        Py_ssize_t parent = (pos - 1) / 2;
        int cmp = PyObject_RichCompareBool(heap[pos].priority, heap[parent].priority, Py_LT);
        if (cmp <= 0)
            break;
        PQ_swap(&heap[pos], &heap[parent]);
        pos = parent;
    }
}

static void
PQ_sift_down(PQEntry *heap, Py_ssize_t size, Py_ssize_t pos)
{
    for (;;) {
        Py_ssize_t smallest = pos;
        Py_ssize_t left = 2 * pos + 1;
        Py_ssize_t right = 2 * pos + 2;
        if (left < size) {
            int cmp = PyObject_RichCompareBool(heap[left].priority, heap[smallest].priority, Py_LT);
            if (cmp > 0)
                smallest = left;
        }
        if (right < size) {
            int cmp = PyObject_RichCompareBool(heap[right].priority, heap[smallest].priority, Py_LT);
            if (cmp > 0)
                smallest = right;
        }
        if (smallest == pos)
            break;
        PQ_swap(&heap[pos], &heap[smallest]);
        pos = smallest;
    }
}

static PyObject *
PQ_put(PriorityQueueObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"item", "priority", "blocking", "timeout", NULL};
    PyObject *item;
    PyObject *priority = NULL;
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Opd", kwlist,
                                     &item, &priority, &blocking, &timeout))
        return NULL;

    if (priority == NULL)
        priority = PyLong_FromLong(0);
    else
        Py_INCREF(priority);

    ct_mutex_lock(&self->mutex);

    if (self->maxsize > 0 && self->size >= self->maxsize) {
        if (!blocking) {
            ct_mutex_unlock(&self->mutex);
            Py_DECREF(priority);
            PyErr_SetNone(QueueFull);
            return NULL;
        }
        /* PQ has no not_full condvar — reject immediately for bounded */
        ct_mutex_unlock(&self->mutex);
        Py_DECREF(priority);
        PyErr_SetNone(QueueFull);
        return NULL;
    }

    /* Grow heap if needed */
    if (self->size >= self->capacity) {
        Py_ssize_t new_cap = self->capacity * 2;
        PQEntry *buf = (PQEntry *)PyMem_Realloc(self->heap, (size_t)new_cap * sizeof(PQEntry));
        if (buf == NULL) {
            ct_mutex_unlock(&self->mutex);
            Py_DECREF(priority);
            return PyErr_NoMemory();
        }
        self->heap = buf;
        self->capacity = new_cap;
    }

    Py_INCREF(item);
    self->heap[self->size].priority = priority;
    self->heap[self->size].item = item;
    self->size++;
    PQ_sift_up(self->heap, self->size - 1);

    ct_cond_signal(&self->not_empty);
    ct_mutex_unlock(&self->mutex);
    atomic_fetch_add_explicit(&self->total_put, 1, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
PQ_put_nowait(PriorityQueueObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"item", "priority", NULL};
    PyObject *item;
    PyObject *priority = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &item, &priority))
        return NULL;

    if (priority == NULL)
        priority = PyLong_FromLong(0);
    else
        Py_INCREF(priority);

    ct_mutex_lock(&self->mutex);
    if (self->maxsize > 0 && self->size >= self->maxsize) {
        ct_mutex_unlock(&self->mutex);
        Py_DECREF(priority);
        PyErr_SetNone(QueueFull);
        return NULL;
    }

    if (self->size >= self->capacity) {
        Py_ssize_t new_cap = self->capacity * 2;
        PQEntry *buf = (PQEntry *)PyMem_Realloc(self->heap, (size_t)new_cap * sizeof(PQEntry));
        if (buf == NULL) {
            ct_mutex_unlock(&self->mutex);
            Py_DECREF(priority);
            return PyErr_NoMemory();
        }
        self->heap = buf;
        self->capacity = new_cap;
    }

    Py_INCREF(item);
    self->heap[self->size].priority = priority;
    self->heap[self->size].item = item;
    self->size++;
    PQ_sift_up(self->heap, self->size - 1);

    ct_cond_signal(&self->not_empty);
    ct_mutex_unlock(&self->mutex);
    atomic_fetch_add_explicit(&self->total_put, 1, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
PQ_get(PriorityQueueObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"blocking", "timeout", NULL};
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pd", kwlist, &blocking, &timeout))
        return NULL;

    ct_mutex_lock(&self->mutex);

    if (self->size == 0) {
        if (!blocking) {
            ct_mutex_unlock(&self->mutex);
            PyErr_SetNone(QueueEmpty);
            return NULL;
        }
        if (timeout < 0) {
            while (self->size == 0)
                ct_cond_wait(&self->not_empty, &self->mutex);
        } else {
            double deadline = ct_time_ms() + timeout * 1000.0;
            while (self->size == 0) {
                double remaining = deadline - ct_time_ms();
                if (remaining <= 0) {
                    ct_mutex_unlock(&self->mutex);
                    PyErr_SetNone(QueueEmpty);
                    return NULL;
                }
                unsigned long ms = (unsigned long)remaining;
                if (ms == 0) ms = 1;
                ct_cond_timedwait_ms(&self->not_empty, &self->mutex, ms);
            }
        }
    }

    PyObject *item = self->heap[0].item;
    Py_DECREF(self->heap[0].priority);
    self->size--;
    if (self->size > 0) {
        self->heap[0] = self->heap[self->size];
        PQ_sift_down(self->heap, self->size, 0);
    }

    ct_mutex_unlock(&self->mutex);
    atomic_fetch_add_explicit(&self->total_get, 1, memory_order_relaxed);
    return item;
}

static PyObject *
PQ_get_nowait(PriorityQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    if (self->size == 0) {
        ct_mutex_unlock(&self->mutex);
        PyErr_SetNone(QueueEmpty);
        return NULL;
    }

    PyObject *item = self->heap[0].item;
    Py_DECREF(self->heap[0].priority);
    self->size--;
    if (self->size > 0) {
        self->heap[0] = self->heap[self->size];
        PQ_sift_down(self->heap, self->size, 0);
    }

    ct_mutex_unlock(&self->mutex);
    atomic_fetch_add_explicit(&self->total_get, 1, memory_order_relaxed);
    return item;
}

static PyObject *
PQ_qsize(PriorityQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    Py_ssize_t s = self->size;
    ct_mutex_unlock(&self->mutex);
    return PyLong_FromSsize_t(s);
}

static PyObject *
PQ_empty(PriorityQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    int e = (self->size == 0);
    ct_mutex_unlock(&self->mutex);
    if (e) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
PQ_full(PriorityQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->maxsize <= 0)
        Py_RETURN_FALSE;
    ct_mutex_lock(&self->mutex);
    int f = (self->size >= self->maxsize);
    ct_mutex_unlock(&self->mutex);
    if (f) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
PQ_stats(PriorityQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    return Py_BuildValue("{s:L, s:L, s:n}",
        "total_put", atomic_load_explicit(&self->total_put, memory_order_relaxed),
        "total_get", atomic_load_explicit(&self->total_get, memory_order_relaxed),
        "current_size", self->size);
}

static PyMethodDef PQ_methods[] = {
    {"put",         (PyCFunction)PQ_put,         METH_VARARGS | METH_KEYWORDS, "Put item with priority"},
    {"put_nowait",  (PyCFunction)PQ_put_nowait,  METH_VARARGS | METH_KEYWORDS, "Put item without blocking"},
    {"get",         (PyCFunction)PQ_get,         METH_VARARGS | METH_KEYWORDS, "Get highest-priority item"},
    {"get_nowait",  (PyCFunction)PQ_get_nowait,  METH_NOARGS,                  "Get item without blocking"},
    {"qsize",       (PyCFunction)PQ_qsize,       METH_NOARGS,                  "Return queue size"},
    {"empty",       (PyCFunction)PQ_empty,       METH_NOARGS,                  "Return True if empty"},
    {"full",        (PyCFunction)PQ_full,        METH_NOARGS,                  "Return True if full"},
    {"stats",       (PyCFunction)PQ_stats,       METH_NOARGS,                  "Get queue statistics"},
    {NULL}
};

static PyTypeObject PriorityQueueType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._queue.PriorityQueue",
    .tp_doc       = "High-performance thread-safe priority queue (min-heap).",
    .tp_basicsize = sizeof(PriorityQueueObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = PQ_new,
    .tp_dealloc   = (destructor)PQ_dealloc,
    .tp_traverse  = (traverseproc)PQ_traverse,
    .tp_clear     = (inquiry)PQ_clear_items,
    .tp_methods   = PQ_methods,
};

/* ================================================================
 * LIFO QUEUE (stack)
 * ================================================================ */

typedef struct LifoNode {
    PyObject          *item;
    struct LifoNode   *next;
} LifoNode;

typedef struct {
    PyObject_HEAD
    ct_mutex_t         mutex;
    ct_cond_t          not_empty;
    ct_cond_t          not_full;
    ct_cond_t          all_done;
    LifoNode          *top;
    Py_ssize_t         size;
    Py_ssize_t         maxsize;
    Py_ssize_t         unfinished;
    atomic_llong       total_put;
    atomic_llong       total_get;
} LifoQueueObject;

static int
Lifo_traverse(LifoQueueObject *self, visitproc visit, void *arg)
{
    LifoNode *n = self->top;
    while (n) { Py_VISIT(n->item); n = n->next; }
    return 0;
}

static int
Lifo_clear_items(LifoQueueObject *self)
{
    LifoNode *n = self->top;
    while (n) {
        LifoNode *next = n->next;
        Py_CLEAR(n->item);
        PyMem_Free(n);
        n = next;
    }
    self->top = NULL;
    self->size = 0;
    return 0;
}

static void
Lifo_dealloc(LifoQueueObject *self)
{
    PyObject_GC_UnTrack(self);
    Lifo_clear_items(self);
    ct_cond_destroy(&self->all_done);
    ct_cond_destroy(&self->not_full);
    ct_cond_destroy(&self->not_empty);
    ct_mutex_destroy(&self->mutex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
Lifo_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"maxsize", NULL};
    Py_ssize_t maxsize = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|n", kwlist, &maxsize))
        return NULL;

    LifoQueueObject *self = (LifoQueueObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    ct_mutex_init(&self->mutex);
    ct_cond_init(&self->not_empty);
    ct_cond_init(&self->not_full);
    ct_cond_init(&self->all_done);
    self->top = NULL;
    self->size = 0;
    self->maxsize = maxsize;
    self->unfinished = 0;
    atomic_init(&self->total_put, 0);
    atomic_init(&self->total_get, 0);

    return (PyObject *)self;
}

static PyObject *
Lifo_put(LifoQueueObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"item", "blocking", "timeout", NULL};
    PyObject *item;
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|pd", kwlist, &item, &blocking, &timeout))
        return NULL;

    Py_INCREF(item);
    ct_mutex_lock(&self->mutex);

    if (self->maxsize > 0 && self->size >= self->maxsize) {
        if (!blocking) {
            ct_mutex_unlock(&self->mutex);
            Py_DECREF(item);
            PyErr_SetNone(QueueFull);
            return NULL;
        }
        if (timeout < 0) {
            while (self->size >= self->maxsize)
                ct_cond_wait(&self->not_full, &self->mutex);
        } else {
            double deadline = ct_time_ms() + timeout * 1000.0;
            while (self->size >= self->maxsize) {
                double remaining = deadline - ct_time_ms();
                if (remaining <= 0) {
                    ct_mutex_unlock(&self->mutex);
                    Py_DECREF(item);
                    PyErr_SetNone(QueueFull);
                    return NULL;
                }
                unsigned long ms = (unsigned long)remaining;
                if (ms == 0) ms = 1;
                ct_cond_timedwait_ms(&self->not_full, &self->mutex, ms);
            }
        }
    }

    LifoNode *node = (LifoNode *)PyMem_Calloc(1, sizeof(LifoNode));
    if (!node) {
        ct_mutex_unlock(&self->mutex);
        Py_DECREF(item);
        return PyErr_NoMemory();
    }
    node->item = item;
    node->next = self->top;
    self->top = node;
    self->size++;
    self->unfinished++;

    ct_cond_signal(&self->not_empty);
    ct_mutex_unlock(&self->mutex);
    atomic_fetch_add_explicit(&self->total_put, 1, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
Lifo_put_nowait(LifoQueueObject *self, PyObject *args)
{
    PyObject *item;
    if (!PyArg_ParseTuple(args, "O", &item))
        return NULL;

    Py_INCREF(item);
    ct_mutex_lock(&self->mutex);

    if (self->maxsize > 0 && self->size >= self->maxsize) {
        ct_mutex_unlock(&self->mutex);
        Py_DECREF(item);
        PyErr_SetNone(QueueFull);
        return NULL;
    }

    LifoNode *node = (LifoNode *)PyMem_Calloc(1, sizeof(LifoNode));
    if (!node) {
        ct_mutex_unlock(&self->mutex);
        Py_DECREF(item);
        return PyErr_NoMemory();
    }
    node->item = item;
    node->next = self->top;
    self->top = node;
    self->size++;
    self->unfinished++;

    ct_cond_signal(&self->not_empty);
    ct_mutex_unlock(&self->mutex);
    atomic_fetch_add_explicit(&self->total_put, 1, memory_order_relaxed);
    Py_RETURN_NONE;
}

static PyObject *
Lifo_get(LifoQueueObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"blocking", "timeout", NULL};
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pd", kwlist, &blocking, &timeout))
        return NULL;

    ct_mutex_lock(&self->mutex);

    if (self->size == 0) {
        if (!blocking) {
            ct_mutex_unlock(&self->mutex);
            PyErr_SetNone(QueueEmpty);
            return NULL;
        }
        if (timeout < 0) {
            while (self->size == 0)
                ct_cond_wait(&self->not_empty, &self->mutex);
        } else {
            double deadline = ct_time_ms() + timeout * 1000.0;
            while (self->size == 0) {
                double remaining = deadline - ct_time_ms();
                if (remaining <= 0) {
                    ct_mutex_unlock(&self->mutex);
                    PyErr_SetNone(QueueEmpty);
                    return NULL;
                }
                unsigned long ms = (unsigned long)remaining;
                if (ms == 0) ms = 1;
                ct_cond_timedwait_ms(&self->not_empty, &self->mutex, ms);
            }
        }
    }

    LifoNode *node = self->top;
    self->top = node->next;
    self->size--;

    if (self->maxsize > 0)
        ct_cond_signal(&self->not_full);

    ct_mutex_unlock(&self->mutex);

    PyObject *item = node->item;
    PyMem_Free(node);
    atomic_fetch_add_explicit(&self->total_get, 1, memory_order_relaxed);
    return item;
}

static PyObject *
Lifo_get_nowait(LifoQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    if (self->size == 0) {
        ct_mutex_unlock(&self->mutex);
        PyErr_SetNone(QueueEmpty);
        return NULL;
    }

    LifoNode *node = self->top;
    self->top = node->next;
    self->size--;
    if (self->maxsize > 0)
        ct_cond_signal(&self->not_full);
    ct_mutex_unlock(&self->mutex);

    PyObject *item = node->item;
    PyMem_Free(node);
    atomic_fetch_add_explicit(&self->total_get, 1, memory_order_relaxed);
    return item;
}

static PyObject *
Lifo_task_done(LifoQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    if (self->unfinished <= 0) {
        ct_mutex_unlock(&self->mutex);
        PyErr_SetString(PyExc_ValueError, "task_done() called too many times");
        return NULL;
    }
    self->unfinished--;
    if (self->unfinished == 0)
        ct_cond_broadcast(&self->all_done);
    ct_mutex_unlock(&self->mutex);
    Py_RETURN_NONE;
}

static PyObject *
Lifo_join(LifoQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    while (self->unfinished > 0)
        ct_cond_wait(&self->all_done, &self->mutex);
    ct_mutex_unlock(&self->mutex);
    Py_RETURN_NONE;
}

static PyObject *
Lifo_qsize(LifoQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    Py_ssize_t s = self->size;
    ct_mutex_unlock(&self->mutex);
    return PyLong_FromSsize_t(s);
}

static PyObject *
Lifo_empty(LifoQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    int e = (self->size == 0);
    ct_mutex_unlock(&self->mutex);
    if (e) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Lifo_full(LifoQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    if (self->maxsize <= 0) Py_RETURN_FALSE;
    ct_mutex_lock(&self->mutex);
    int f = (self->size >= self->maxsize);
    ct_mutex_unlock(&self->mutex);
    if (f) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject *
Lifo_stats(LifoQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    return Py_BuildValue("{s:L, s:L, s:n}",
        "total_put", atomic_load_explicit(&self->total_put, memory_order_relaxed),
        "total_get", atomic_load_explicit(&self->total_get, memory_order_relaxed),
        "current_size", self->size);
}

static PyMethodDef Lifo_methods[] = {
    {"put",         (PyCFunction)Lifo_put,         METH_VARARGS | METH_KEYWORDS, "Push item onto the stack"},
    {"put_nowait",  (PyCFunction)Lifo_put_nowait,  METH_VARARGS,                 "Push item without blocking"},
    {"get",         (PyCFunction)Lifo_get,         METH_VARARGS | METH_KEYWORDS, "Pop item from the stack"},
    {"get_nowait",  (PyCFunction)Lifo_get_nowait,  METH_NOARGS,                  "Pop item without blocking"},
    {"task_done",   (PyCFunction)Lifo_task_done,   METH_NOARGS,                  "Signal that a task is complete"},
    {"join",        (PyCFunction)Lifo_join,         METH_NOARGS,                  "Block until all tasks are done"},
    {"qsize",       (PyCFunction)Lifo_qsize,       METH_NOARGS,                  "Return queue size"},
    {"empty",       (PyCFunction)Lifo_empty,       METH_NOARGS,                  "Return True if empty"},
    {"full",        (PyCFunction)Lifo_full,        METH_NOARGS,                  "Return True if full"},
    {"stats",       (PyCFunction)Lifo_stats,       METH_NOARGS,                  "Get queue statistics"},
    {NULL}
};

static PyTypeObject LifoQueueType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._queue.LifoQueue",
    .tp_doc       = "High-performance thread-safe LIFO queue (stack).",
    .tp_basicsize = sizeof(LifoQueueObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = Lifo_new,
    .tp_dealloc   = (destructor)Lifo_dealloc,
    .tp_traverse  = (traverseproc)Lifo_traverse,
    .tp_clear     = (inquiry)Lifo_clear_items,
    .tp_methods   = Lifo_methods,
};

/* ================================================================
 * SIMPLE QUEUE (unbounded FIFO, no task tracking, minimal overhead)
 * ================================================================ */

typedef struct SimpleNode {
    PyObject            *item;
    struct SimpleNode   *next;
} SimpleNode;

typedef struct {
    PyObject_HEAD
    ct_mutex_t     mutex;
    ct_cond_t      not_empty;
    SimpleNode    *head;
    SimpleNode    *tail;
    Py_ssize_t     size;
} SimpleQueueObject;

static int
SQ_traverse(SimpleQueueObject *self, visitproc visit, void *arg)
{
    SimpleNode *n = self->head;
    while (n) { Py_VISIT(n->item); n = n->next; }
    return 0;
}

static int
SQ_clear_items(SimpleQueueObject *self)
{
    SimpleNode *n = self->head;
    while (n) {
        SimpleNode *next = n->next;
        Py_CLEAR(n->item);
        PyMem_Free(n);
        n = next;
    }
    self->head = self->tail = NULL;
    self->size = 0;
    return 0;
}

static void
SQ_dealloc(SimpleQueueObject *self)
{
    PyObject_GC_UnTrack(self);
    SQ_clear_items(self);
    ct_cond_destroy(&self->not_empty);
    ct_mutex_destroy(&self->mutex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
SQ_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    SimpleQueueObject *self = (SimpleQueueObject *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    ct_mutex_init(&self->mutex);
    ct_cond_init(&self->not_empty);
    self->head = self->tail = NULL;
    self->size = 0;

    return (PyObject *)self;
}

static PyObject *
SQ_put(SimpleQueueObject *self, PyObject *args)
{
    PyObject *item;
    if (!PyArg_ParseTuple(args, "O", &item))
        return NULL;

    Py_INCREF(item);

    SimpleNode *node = (SimpleNode *)PyMem_Calloc(1, sizeof(SimpleNode));
    if (!node) { Py_DECREF(item); return PyErr_NoMemory(); }
    node->item = item;
    node->next = NULL;

    ct_mutex_lock(&self->mutex);
    if (self->tail)
        self->tail->next = node;
    else
        self->head = node;
    self->tail = node;
    self->size++;
    ct_cond_signal(&self->not_empty);
    ct_mutex_unlock(&self->mutex);
    Py_RETURN_NONE;
}

static PyObject *
SQ_get(SimpleQueueObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"blocking", "timeout", NULL};
    int blocking = 1;
    double timeout = -1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|pd", kwlist, &blocking, &timeout))
        return NULL;

    ct_mutex_lock(&self->mutex);

    if (self->size == 0) {
        if (!blocking) {
            ct_mutex_unlock(&self->mutex);
            PyErr_SetNone(QueueEmpty);
            return NULL;
        }
        if (timeout < 0) {
            while (self->size == 0)
                ct_cond_wait(&self->not_empty, &self->mutex);
        } else {
            double deadline = ct_time_ms() + timeout * 1000.0;
            while (self->size == 0) {
                double remaining = deadline - ct_time_ms();
                if (remaining <= 0) {
                    ct_mutex_unlock(&self->mutex);
                    PyErr_SetNone(QueueEmpty);
                    return NULL;
                }
                unsigned long ms = (unsigned long)remaining;
                if (ms == 0) ms = 1;
                ct_cond_timedwait_ms(&self->not_empty, &self->mutex, ms);
            }
        }
    }

    SimpleNode *node = self->head;
    self->head = node->next;
    if (!self->head) self->tail = NULL;
    self->size--;
    ct_mutex_unlock(&self->mutex);

    PyObject *item = node->item;
    PyMem_Free(node);
    return item;
}

static PyObject *
SQ_get_nowait(SimpleQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    if (self->size == 0) {
        ct_mutex_unlock(&self->mutex);
        PyErr_SetNone(QueueEmpty);
        return NULL;
    }

    SimpleNode *node = self->head;
    self->head = node->next;
    if (!self->head) self->tail = NULL;
    self->size--;
    ct_mutex_unlock(&self->mutex);

    PyObject *item = node->item;
    PyMem_Free(node);
    return item;
}

static PyObject *
SQ_put_nowait(SimpleQueueObject *self, PyObject *args)
{
    return SQ_put(self, args);
}

static PyObject *
SQ_qsize(SimpleQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    Py_ssize_t s = self->size;
    ct_mutex_unlock(&self->mutex);
    return PyLong_FromSsize_t(s);
}

static PyObject *
SQ_empty(SimpleQueueObject *self, PyObject *Py_UNUSED(ignored))
{
    ct_mutex_lock(&self->mutex);
    int e = (self->size == 0);
    ct_mutex_unlock(&self->mutex);
    if (e) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyMethodDef SQ_methods[] = {
    {"put",         (PyCFunction)SQ_put,         METH_VARARGS,                 "Put an item (never blocks)"},
    {"put_nowait",  (PyCFunction)SQ_put_nowait,  METH_VARARGS,                 "Put an item (alias for put)"},
    {"get",         (PyCFunction)SQ_get,         METH_VARARGS | METH_KEYWORDS, "Get an item"},
    {"get_nowait",  (PyCFunction)SQ_get_nowait,  METH_NOARGS,                  "Get an item without blocking"},
    {"qsize",       (PyCFunction)SQ_qsize,       METH_NOARGS,                  "Return queue size"},
    {"empty",       (PyCFunction)SQ_empty,       METH_NOARGS,                  "Return True if empty"},
    {NULL}
};

static PyTypeObject SimpleQueueType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "cthreading._queue.SimpleQueue",
    .tp_doc       = "Unbounded FIFO queue with no task tracking — minimal overhead.",
    .tp_basicsize = sizeof(SimpleQueueObject),
    .tp_flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,
    .tp_new       = SQ_new,
    .tp_dealloc   = (destructor)SQ_dealloc,
    .tp_traverse  = (traverseproc)SQ_traverse,
    .tp_clear     = (inquiry)SQ_clear_items,
    .tp_methods   = SQ_methods,
};

/* ================================================================
 * MODULE DEFINITION
 * ================================================================ */

static PyMethodDef queue_module_methods[] = {
    {NULL, NULL, 0, NULL},
};

static PyModuleDef queue_module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name    = "cthreading._queue",
    .m_doc     = "cthreading C core — Queue, PriorityQueue, LifoQueue, SimpleQueue",
    .m_size    = -1,
    .m_methods = queue_module_methods,
};

PyMODINIT_FUNC
PyInit__queue(void)
{
    PyObject *m;

    if (PyType_Ready(&QueueType) < 0)         return NULL;
    if (PyType_Ready(&PriorityQueueType) < 0) return NULL;
    if (PyType_Ready(&LifoQueueType) < 0)     return NULL;
    if (PyType_Ready(&SimpleQueueType) < 0)   return NULL;

    m = PyModule_Create(&queue_module_def);
    if (m == NULL)
        return NULL;

    /* Create Empty and Full exceptions */
    QueueEmpty = PyErr_NewException("cthreading._queue.Empty", NULL, NULL);
    if (!QueueEmpty) { Py_DECREF(m); return NULL; }
    Py_INCREF(QueueEmpty);
    if (PyModule_AddObject(m, "Empty", QueueEmpty) < 0) {
        Py_DECREF(QueueEmpty); Py_DECREF(m); return NULL;
    }

    QueueFull = PyErr_NewException("cthreading._queue.Full", NULL, NULL);
    if (!QueueFull) { Py_DECREF(m); return NULL; }
    Py_INCREF(QueueFull);
    if (PyModule_AddObject(m, "Full", QueueFull) < 0) {
        Py_DECREF(QueueFull); Py_DECREF(m); return NULL;
    }

#define ADD_TYPE(name, typeobj)                             \
    Py_INCREF(&(typeobj));                                 \
    if (PyModule_AddObject(m, name, (PyObject *)&(typeobj)) < 0) { \
        Py_DECREF(&(typeobj));                             \
        Py_DECREF(m);                                     \
        return NULL;                                       \
    }

    ADD_TYPE("Queue",         QueueType);
    ADD_TYPE("PriorityQueue", PriorityQueueType);
    ADD_TYPE("LifoQueue",     LifoQueueType);
    ADD_TYPE("SimpleQueue",   SimpleQueueType);

#undef ADD_TYPE

#ifdef Py_MOD_GIL_NOT_USED
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    return m;
}
