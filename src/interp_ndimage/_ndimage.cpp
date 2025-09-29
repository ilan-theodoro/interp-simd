#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" {
#define PY_ARRAY_UNIQUE_SYMBOL INTERP_NDIMAGE_ARRAY_API
#include <numpy/arrayobject.h>
#include "nd_image_interpolation.h"
}

namespace py = pybind11;

namespace {

py::object call_no_kwargs(PyObject *(*callable)(PyObject *), py::args args, py::kwargs kwargs)
{
    if (!kwargs.empty()) {
        throw py::type_error("keyword arguments are not supported by the low-level ndimage bindings");
    }
    PyObject *result = callable(args.ptr());
    if (!result) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(result);
}

}  // namespace

PYBIND11_MODULE(_ndimage, m)
{
    if (_import_array() < 0) {
        throw py::error_already_set();
    }

    m.doc() = "Pybind11 bindings wrapping SciPy ndimage interpolation C kernels";

    m.def("spline_filter1d",
          [](py::args args, py::kwargs kwargs) {
              return call_no_kwargs(interp_ndimage_spline_filter1d, args, kwargs);
          });

    m.def("geometric_transform",
          [](py::args args, py::kwargs kwargs) {
              return call_no_kwargs(interp_ndimage_geometric_transform, args, kwargs);
          });

    m.def("zoom_shift",
          [](py::args args, py::kwargs kwargs) {
              return call_no_kwargs(interp_ndimage_zoom_shift, args, kwargs);
          });
}
