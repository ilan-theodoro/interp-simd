/* Derived from SciPy's nd_image.c, trimmed to interpolation routines only. */

#include "nd_image.h"
#include "ni_support.h"
#include "ni_interpolation.h"
#include "ccallback.h"

#include <stdlib.h>
#include <string.h>

int interp_ndimage_initialize_numpy(void)
{
    import_array1(-1);
    return 0;
}


typedef struct {
    PyObject *extra_arguments;
    PyObject *extra_keywords;
} NI_PythonCallbackData;

/* NumPy array helpers */
static int NI_ObjectToInputArray(PyObject *object, PyArrayObject **array)
{
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED;
    *array = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    return *array != NULL;
}

static int NI_ObjectToOptionalInputArray(PyObject *object, PyArrayObject **array)
{
    if (object == Py_None) {
        *array = NULL;
        return 1;
    }
    return NI_ObjectToInputArray(object, array);
}

static int NI_ObjectToOutputArray(PyObject *object, PyArrayObject **array)
{
    int flags = NPY_ARRAY_BEHAVED_NS | NPY_ARRAY_WRITEBACKIFCOPY;
    if (PyArray_Check(object) && !PyArray_ISWRITEABLE((PyArrayObject *)object)) {
        PyErr_SetString(PyExc_ValueError, "output array is read-only.");
        return 0;
    }
    *array = (PyArrayObject *)PyArray_CheckFromAny(object, NULL, 0, 0, flags, NULL);
    return *array != NULL;
}

/* Callback adapter used by geometric_transform */
static int Py_Map(npy_intp *ocoor, double *icoor, int orank, int irank, void *data)
{
    PyObject *coors = NULL, *rets = NULL, *args = NULL, *tmp = NULL;
    npy_intp ii;
    ccallback_t *callback = (ccallback_t *)data;
    NI_PythonCallbackData *cbdata = (NI_PythonCallbackData *)callback->info_p;

    coors = PyTuple_New(orank);
    if (!coors)
        goto exit;
    for (ii = 0; ii < orank; ii++) {
        PyTuple_SetItem(coors, ii, PyLong_FromSsize_t(ocoor[ii]));
        if (PyErr_Occurred())
            goto exit;
    }
    tmp = Py_BuildValue("(O)", coors);
    if (!tmp)
        goto exit;
    args = PySequence_Concat(tmp, cbdata->extra_arguments);
    if (!args)
        goto exit;
    rets = PyObject_Call(callback->py_function, args, cbdata->extra_keywords);
    if (!rets)
        goto exit;
    for (ii = 0; ii < irank; ii++) {
        icoor[ii] = PyFloat_AsDouble(PyTuple_GetItem(rets, ii));
        if (PyErr_Occurred())
            goto exit;
    }
exit:
    Py_XDECREF(coors);
    Py_XDECREF(tmp);
    Py_XDECREF(rets);
    Py_XDECREF(args);
    return PyErr_Occurred() ? 0 : 1;
}

static PyObject *Py_SplineFilter1D(PyObject *NPY_UNUSED(obj), PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL;
    int order, axis, mode;

    if (!PyArg_ParseTuple(args, "O&iiO&i",
                          NI_ObjectToInputArray, &input,
                          &order, &axis,
                          NI_ObjectToOutputArray, &output,
                          &mode))
        goto exit;

    NI_SplineFilter1D(input, order, axis, (NI_ExtendMode)mode, output);
    PyArray_ResolveWritebackIfCopy(output);

exit:
    Py_XDECREF(input);
    Py_XDECREF(output);
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_GeometricTransform(PyObject *NPY_UNUSED(obj), PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL;
    PyArrayObject *coordinates = NULL, *matrix = NULL, *shift = NULL;
    PyObject *fnc = NULL, *extra_arguments = NULL, *extra_keywords = NULL;
    int mode, order, nprepad;
    double cval;
    int (*func)(npy_intp*, double*, int, int, void*) = NULL;
    void *data = NULL;
    NI_PythonCallbackData cbdata;
    ccallback_t callback;
    static ccallback_signature_t callback_signatures[] = {
        {"int (intptr_t *, double *, int, int, void *)"},
        {"int (npy_intp *, double *, int, int, void *)"},
#if NPY_SIZEOF_INTP == NPY_SIZEOF_SHORT
        {"int (short *, double *, int, int, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_INT
        {"int (int *, double *, int, int, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_LONG
        {"int (long *, double *, int, int, void *)"},
#endif
#if NPY_SIZEOF_INTP == NPY_SIZEOF_LONGLONG
        {"int (long long *, double *, int, int, void *)"},
#endif
        {NULL}
    };

    callback.py_function = NULL;
    callback.c_function = NULL;

    if (!PyArg_ParseTuple(args, "O&OO&O&O&O&iidiOO",
                          NI_ObjectToInputArray, &input,
                          &fnc,
                          NI_ObjectToOptionalInputArray, &coordinates,
                          NI_ObjectToOptionalInputArray, &matrix,
                          NI_ObjectToOptionalInputArray, &shift,
                          NI_ObjectToOutputArray, &output,
                          &order, &mode, &cval, &nprepad,
                          &extra_arguments, &extra_keywords))
        goto exit;

    if (fnc != Py_None) {
        if (!PyTuple_Check(extra_arguments)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "extra_arguments must be a tuple");
            goto exit;
        }
        if (!PyDict_Check(extra_keywords)) {
            PyErr_SetString(PyExc_RuntimeError,
                            "extra_keywords must be a dictionary");
            goto exit;
        }
        if (PyCapsule_CheckExact(fnc) && PyCapsule_GetName(fnc) == NULL) {
            func = (int (*)(npy_intp*, double*, int, int, void*))PyCapsule_GetPointer(fnc, NULL);
            data = PyCapsule_GetContext(fnc);
        } else {
            int ret;

            ret = ccallback_prepare(&callback, callback_signatures, fnc,
                                    CCALLBACK_DEFAULTS);
            if (ret == -1) {
                goto exit;
            }

            if (callback.py_function != NULL) {
                cbdata.extra_arguments = extra_arguments;
                cbdata.extra_keywords = extra_keywords;
                callback.info_p = (void *)&cbdata;
                func = Py_Map;
                data = (void *)&callback;
            }
            else {
                func = (int (*)(npy_intp*, double*, int, int, void*))callback.c_function;
                data = callback.user_data;
            }
        }
    }

    NI_GeometricTransform(input, func, data, matrix, shift, coordinates,
                          output, order, (NI_ExtendMode)mode, cval, nprepad);
    PyArray_ResolveWritebackIfCopy(output);

exit:
    if (callback.py_function != NULL || callback.c_function != NULL) {
        ccallback_release(&callback);
    }
    Py_XDECREF(input);
    Py_XDECREF(output);
    Py_XDECREF(coordinates);
    Py_XDECREF(matrix);
    Py_XDECREF(shift);
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

static PyObject *Py_ZoomShift(PyObject *NPY_UNUSED(obj), PyObject *args)
{
    PyArrayObject *input = NULL, *output = NULL, *shift = NULL;
    PyArrayObject *zoom = NULL;
    int mode, order, nprepad, grid_mode;
    double cval;

    if (!PyArg_ParseTuple(args, "O&O&O&O&iidii",
                          NI_ObjectToInputArray, &input,
                          NI_ObjectToOptionalInputArray, &zoom,
                          NI_ObjectToOptionalInputArray, &shift,
                          NI_ObjectToOutputArray, &output,
                          &order, &mode, &cval, &nprepad, &grid_mode))
        goto exit;

    NI_ZoomShift(input, zoom, shift, output, order, (NI_ExtendMode)mode, cval,
                 nprepad, grid_mode);
    PyArray_ResolveWritebackIfCopy(output);

exit:
    Py_XDECREF(input);
    Py_XDECREF(shift);
    Py_XDECREF(zoom);
    Py_XDECREF(output);
    return PyErr_Occurred() ? NULL : Py_BuildValue("");
}

PyObject *interp_ndimage_spline_filter1d(PyObject *args)
{
    return Py_SplineFilter1D(NULL, args);
}

PyObject *interp_ndimage_geometric_transform(PyObject *args)
{
    return Py_GeometricTransform(NULL, args);
}

PyObject *interp_ndimage_zoom_shift(PyObject *args)
{
    return Py_ZoomShift(NULL, args);
}
