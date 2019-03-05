//https://docs.scipy.org/doc/numpy/reference/c-api.array.html?highlight=import_array
#define PY_ARRAY_UNIQUE_SYMBOL MY_BIND_C
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "fastvio_common.h"
#include "fastvio.h"


char hellofunc_docs[] = "Hello world description.";
char heymanfunc_docs[] = "Echo your name and passed number.";
char addfunc_docs[] = "Add two numbers function.";

// METH_NOARGS
static PyMethodDef fastvio_funcs[] = {
	{	"open",
		(PyCFunction)fastvio_open,
		METH_VARARGS | METH_KEYWORDS,
		"Open a video file. Return pointer to its ffmpeg context."},
    {	"close",
		(PyCFunction)fastvio_close,
		METH_VARARGS,
		"Close a fastvio handle."},
    {	"grab_frame",
		(PyCFunction)fastvio_grab_frame,
		METH_VARARGS | METH_KEYWORDS,
		"Grab a frame to numpy array."},
    {	"seek",
		(PyCFunction)fastvio_seek,
		METH_VARARGS,
		"Seek to pts."},
    {	"get_duration",
		(PyCFunction)fastvio_get_duration,
		METH_VARARGS,
		"Get the duration of video, in av_timebase"},
    {	"print_dbg",
		(PyCFunction)fastvio_print_dbg,
		METH_VARARGS,
		"Print debugging info."},
		
	{	NULL, NULL, 0, NULL}
};

char fastviomod_docs[] = "This is hello world module.";

static struct PyModuleDef fastvio_mod = {
	PyModuleDef_HEAD_INIT,
	"fastvio",
	fastviomod_docs,
	-1,
	fastvio_funcs,
	NULL,
	NULL,
	NULL,
	NULL
};

PyMODINIT_FUNC PyInit_fastvio(void) {
	PyObject *self;
	self = PyModule_Create(&fastvio_mod);
	if (self == NULL)
        return NULL;
    MyError = PyErr_NewException("fastvio.error", NULL, NULL);
    Py_INCREF(MyError);
    PyModule_AddObject(self, "error", MyError);
    
    import_array();
    init_ffmpeg();
    return self;
}
