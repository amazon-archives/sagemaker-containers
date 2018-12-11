// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the 'License'). You
// may not use this file except in compliance with the License. A copy of
// the License is located at
//
//     http://aws.amazon.com/apache2.0/
//
// or in the 'license' file accompanying this file. This file is
// distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
// ANY KIND, either express or implied. See the License for the specific
// language governing permissions and limitations under the License.

#include <Python.h>
#include <stdlib.h>


int gethostname(char *name, size_t len)
{
  const char *val = getenv("SM_CURRENT_HOST");

  strncpy(name, val, len);
  return 0;
}


static PyObject* gethostname_call(PyObject* self, PyObject* args) {
    long unsigned command;
    char name[40];

    if (!PyArg_ParseTuple(args, "k", &command)) {
        return NULL;
    }

    gethostname(name, command);
    printf("%s", name);

    return Py_BuildValue("s", name);
}

static PyMethodDef GethostnameMethods[] = {
    {
        "call",
        gethostname_call,
        METH_VARARGS,
    },
    {NULL, NULL, 0, NULL},  // sentinel
};

#if PY_MAJOR_VERSION >= 3
static PyModuleDef gethostnamemodule = {
    PyModuleDef_HEAD_INIT,
    "gethostname",
    "Returns the value of $SM_CURRENT_HOST",
    -1,
    GethostnameMethods,
};

PyMODINIT_FUNC PyInit_gethostname() {
    return PyModule_Create(&gethostnamemodule);
}
#else
PyMODINIT_FUNC inithostnamemodule() {
    PyObject* module;

    module = Py_InitModule3(
        "gethostname", GethostnameMethods, "Returns the value of $SM_CURRENT_HOST");
}
#endif