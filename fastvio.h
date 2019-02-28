#ifndef __LIBFASTVIO_H__
#define __LIBFASTVIO_H__

#include <Python.h>

void init_ffmpeg();
PyObject * fastvio_open(PyObject *, PyObject *, PyObject *);
PyObject * fastvio_close(PyObject *, PyObject *);
PyObject * fastvio_grab_frame(PyObject *, PyObject *);
PyObject * fastvio_seek(PyObject *, PyObject *);
PyObject * fastvio_get_duration(PyObject *, PyObject *);
PyObject * fastvio_print_dbg(PyObject *, PyObject *);

#endif
