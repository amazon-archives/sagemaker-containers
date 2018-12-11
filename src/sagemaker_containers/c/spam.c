#include <Python.h>
#include <stdlib.h>


static PyObject* SpamError;

static PyObject* spam_system(PyObject* self, PyObject* args) {
    const char* command;
    int status;

    if (!PyArg_ParseTuple(args, "s", &command)) {
        return NULL;
    }
    status = system(command);
    if (!WIFEXITED(status)) {
        return Py_BuildValue("i", -1);
    }
    return Py_BuildValue("i", WEXITSTATUS(status));
}

static PyObject* spam_check_system(PyObject* self, PyObject* args) {
    const char* command;
    int status;
    int exit_status;

    if (!PyArg_ParseTuple(args, "s", &command)) {
        return NULL;
    }
    status = system(command);
    if (!WIFEXITED(status)) {
        return PyErr_Format(SpamError, "Command could not be executed");
    }
    exit_status = WEXITSTATUS(status);
    if (exit_status != 0) {
        return PyErr_Format(
            SpamError, "Command returned non-zero exit status %d", exit_status);
    }
    Py_RETURN_NONE;
}

static PyMethodDef SpamMethods[] = {
    {
        "system",
        spam_system,
        METH_VARARGS,
        "Execute a system command and return its exit status.\n\n"
        "If the command cannot be executed, return -1.",
    },
    {
        "check_system",
        spam_check_system,
        METH_VARARGS,
        "Execute a system command.\n\n"
        "If the command cannot be executed or returns a non-zero exit status, "
        "raise Error.",
    },
    {NULL, NULL, 0, NULL},  // sentinel
};

#if PY_MAJOR_VERSION >= 3
static PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",
    "An example Python C extension module.",
    -1,
    SpamMethods,
};

PyMODINIT_FUNC PyInit_spam() {
    PyObject* module;

    module = PyModule_Create(&spammodule);
    if (module == NULL) {
        return NULL;
    }
    SpamError = PyErr_NewException("spam.Error", NULL, NULL);
    Py_INCREF(SpamError);
    PyModule_AddObject(module, "Error", SpamError);
    return module;
}
#else
PyMODINIT_FUNC initspam() {
    PyObject* module;

    module = Py_InitModule3(
        "spam", SpamMethods, "An example Python C extension module.");
    if (module == NULL) {
        return;
    }
    SpamError = PyErr_NewException((char*)"spam.Error", NULL, NULL);
    Py_INCREF(SpamError);
    PyModule_AddObject(module, "Error", SpamError);
}
#endif