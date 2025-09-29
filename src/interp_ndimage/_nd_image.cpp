#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" {
#define NO_IMPORT_ARRAY
#include "nd_image.h"
#include "nd_image_interpolation.h"
int interp_ndimage_initialize_numpy(void);
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

PYBIND11_MODULE(_nd_image, m)
{
    if (PyArray_API == NULL) {
        if (interp_ndimage_initialize_numpy() < 0) {
            throw py::error_already_set();
        }
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
