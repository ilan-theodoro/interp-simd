#ifndef INTERP_NDIMAGE_INTERPOLATION_H
#define INTERP_NDIMAGE_INTERPOLATION_H

#include <Python.h>

PyObject *interp_ndimage_spline_filter1d(PyObject *args);
PyObject *interp_ndimage_geometric_transform(PyObject *args);
PyObject *interp_ndimage_zoom_shift(PyObject *args);

#endif /* INTERP_NDIMAGE_INTERPOLATION_H */
