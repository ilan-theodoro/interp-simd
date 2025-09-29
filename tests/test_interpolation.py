import sys
import warnings

import numpy as np
from .array_api_stub import (
    _asarray, assert_array_almost_equal,
    is_jax, np_compat,
    xp_assert_equal, xp_assert_close,
    make_xp_test_case,
)

import pytest
from pytest import raises as assert_raises
import interp_ndimage as ndimage

try:
    import scipy.ndimage as _scipy_ndimage
except ImportError:  # pragma: no cover - SciPy required for reference checks
    _scipy_ndimage = None

from . import types

SUPPORTED_ORDERS = [0, 1]

np.set_printoptions(legacy='1.25')


def require_scipy():
    if _scipy_ndimage is None:
        pytest.skip("SciPy not available for reference comparison")


def scipy_expected(func_name, xp, *args, **kwargs):
    require_scipy()
    np_args = [np.asarray(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
    np_kwargs = {k: (np.asarray(v) if isinstance(v, np.ndarray) else v)
                 for k, v in kwargs.items()}
    result = getattr(_scipy_ndimage, func_name)(*np_args, **np_kwargs)
    return xp.asarray(result)


def compare_to_scipy(func_name, xp, *args, comparator=xp_assert_close,
                     comparator_kwargs=None, **kwargs):
    expected = scipy_expected(func_name, xp, *args, **kwargs)
    result = getattr(ndimage, func_name)(*args, **kwargs)
    if comparator_kwargs is None:
        comparator_kwargs = {}
    comparator(xp.asarray(result), expected, **comparator_kwargs)
    return result, expected


def skip_xp_backends(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


def xfail_xp_backends(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


eps = 1e-12

ndimage_to_numpy_mode = {
    'mirror': 'reflect',
    'reflect': 'symmetric',
    'grid-mirror': 'symmetric',
    'grid-wrap': 'wrap',
    'nearest': 'edge',
    'grid-constant': 'constant',
}


class TestBoundaries:

    @make_xp_test_case(ndimage.geometric_transform)
    @pytest.mark.parametrize(
        'mode',
        ['nearest', 'wrap', 'grid-wrap', 'mirror', 'reflect', 'constant', 'grid-constant']
    )
    def test_boundaries(self, mode, xp):
        def shift(x):
            return (x[0] + 0.5,)

        data = xp.asarray([1, 2, 3, 4.])
        compare_to_scipy('geometric_transform', xp, data, shift,
                         cval=-1, mode=mode, output_shape=(7,), order=1,
                         comparator=xp_assert_equal)

    @make_xp_test_case(ndimage.geometric_transform)
    @pytest.mark.parametrize(
        'mode',
        ['nearest', 'wrap', 'grid-wrap', 'mirror', 'reflect', 'constant', 'grid-constant']
    )
    def test_boundaries2(self, mode, xp):
        def shift(x):
            return (x[0] - 0.9,)

        data = xp.asarray([1, 2, 3, 4])
        compare_to_scipy('geometric_transform', xp, data, shift,
                         cval=-1, mode=mode, output_shape=(4,), order=1,
                         comparator=xp_assert_equal)

    @make_xp_test_case(ndimage.map_coordinates)
    @pytest.mark.parametrize('mode', ['mirror', 'reflect', 'grid-mirror',
                                      'grid-wrap', 'grid-constant',
                                      'nearest'])
    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_boundary_spline_accuracy(self, mode, order, xp):
        """Tests based on examples from gh-2640"""
        if (is_jax(xp) and
            (mode not in ['mirror', 'reflect', 'constant', 'wrap', 'nearest']
             or order > 1)
        ):
            pytest.xfail("Jax does not support grid- modes or order > 1")

        require_scipy()
        np_data = np.arange(-6, 7, dtype=np.float64)
        data = xp.asarray(np_data)
        coords = xp.asarray(np.linspace(-8, 15, num=1000))[xp.newaxis, ...]

        atol = 1e-5 if mode == 'grid-constant' else 1e-12
        compare_to_scipy('map_coordinates', xp, data, coords, order=order,
                         mode=mode, comparator=xp_assert_close,
                         comparator_kwargs={'rtol': 1e-7, 'atol': atol})


@pytest.mark.skip(reason="Spline filters with order > 1 are not implemented in interp-ndimage")
@make_xp_test_case(ndimage.spline_filter)
@pytest.mark.parametrize('order', SUPPORTED_ORDERS)
@pytest.mark.parametrize('dtype', types)
class TestSpline:

    def test_spline01(self, dtype, order, xp):
        dtype = getattr(xp, dtype)
        data = xp.ones([], dtype=dtype)
        out = ndimage.spline_filter(data, order=order)
        assert out == xp.asarray(1, dtype=out.dtype)

    def test_spline02(self, dtype, order, xp):
        dtype = getattr(xp, dtype)
        data = xp.asarray([1], dtype=dtype)
        out = ndimage.spline_filter(data, order=order)
        assert_array_almost_equal(out, xp.asarray([1]))

    @skip_xp_backends(np_only=True, exceptions=["cupy"],
                      reason='output=dtype is numpy-specific')
    def test_spline03(self, dtype, order, xp):
        dtype = getattr(xp, dtype)
        data = xp.ones([], dtype=dtype)
        out = ndimage.spline_filter(data, order, output=dtype)
        assert out == xp.asarray(1, dtype=out.dtype)

    def test_spline04(self, dtype, order, xp):
        dtype = getattr(xp, dtype)
        data = xp.ones([4], dtype=dtype)
        out = ndimage.spline_filter(data, order)
        assert_array_almost_equal(out, xp.asarray([1, 1, 1, 1]))

    def test_spline05(self, dtype, order, xp):
        dtype = getattr(xp, dtype)
        data = xp.ones([4, 4], dtype=dtype)
        out = ndimage.spline_filter(data, order=order)
        expected = xp.asarray([[1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1],
                               [1, 1, 1, 1]])
        assert_array_almost_equal(out, expected)


@make_xp_test_case(ndimage.geometric_transform)
@pytest.mark.parametrize('order', SUPPORTED_ORDERS)
class TestGeometricTransform:

    def test_geometric_transform01(self, order, xp):
        data = xp.asarray([1])

        def mapping(x):
            return x

        compare_to_scipy('geometric_transform', xp, data, mapping, data.shape,
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform02(self, order, xp):
        data = xp.ones([4])

        def mapping(x):
            return x

        compare_to_scipy('geometric_transform', xp, data, mapping, data.shape,
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform03(self, order, xp):
        data = xp.ones([4])

        def mapping(x):
            return (x[0] - 1,)

        compare_to_scipy('geometric_transform', xp, data, mapping, data.shape,
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform04(self, order, xp):
        data = xp.asarray([4, 1, 3, 2])

        def mapping(x):
            return (x[0] - 1,)

        compare_to_scipy('geometric_transform', xp, data, mapping, data.shape,
                         order=order, comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('dtype', ["float64", "complex128"])
    def test_geometric_transform05(self, order, dtype, xp):
        dtype = getattr(xp, dtype)
        data = xp.asarray([[1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1]], dtype=dtype)

        if xp.isdtype(data.dtype, 'complex floating'):
            data -= 1j * data

        def mapping(x):
            return (x[0], x[1] - 1)

        compare_to_scipy('geometric_transform', xp, data, mapping, data.shape,
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform06(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])

        def mapping(x):
            return (x[0], x[1] - 1)

        compare_to_scipy('geometric_transform', xp, data, mapping, data.shape,
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform07(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])

        def mapping(x):
            return (x[0] - 1, x[1])

        compare_to_scipy('geometric_transform', xp, data, mapping, data.shape,
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform08(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])

        def mapping(x):
            return (x[0] - 1, x[1] - 1)

        compare_to_scipy('geometric_transform', xp, data, mapping, data.shape,
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform10(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])

        def mapping(x):
            return (x[0] - 1, x[1] - 1)

        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        compare_to_scipy('geometric_transform', xp, filtered, mapping,
                         data.shape, order=order, prefilter=False,
                         comparator=assert_array_almost_equal)

    def test_geometric_transform13(self, order, xp):
        data = xp.ones([2], dtype=xp.float64)

        def mapping(x):
            return (x[0] // 2,)

        compare_to_scipy('geometric_transform', xp, data, mapping, [4],
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform14(self, order, xp):
        data = xp.asarray([1, 5, 2, 6, 3, 7, 4, 4])

        def mapping(x):
            return (2 * x[0],)

        compare_to_scipy('geometric_transform', xp, data, mapping, [4],
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform15(self, order, xp):
        data = xp.asarray([1, 2, 3, 4])

        def mapping(x):
            return (x[0] / 2,)

        result, expected = compare_to_scipy('geometric_transform', xp, data,
                                            mapping, [8], order=order,
                                            comparator=assert_array_almost_equal)
        assert_array_almost_equal(result[::2], expected[::2])

    def test_geometric_transform16(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9.0, 10, 11, 12]]
        data = xp.asarray(data)

        def mapping(x):
            return (x[0], x[1] * 2)

        compare_to_scipy('geometric_transform', xp, data, mapping, (3, 2),
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform17(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)

        def mapping(x):
            return (x[0] * 2, x[1])

        compare_to_scipy('geometric_transform', xp, data, mapping, (1, 4),
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform18(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)

        def mapping(x):
            return (x[0] * 2, x[1] * 2)

        compare_to_scipy('geometric_transform', xp, data, mapping, (1, 2),
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform19(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)

        def mapping(x):
            return (x[0], x[1] / 2)

        result, expected = compare_to_scipy('geometric_transform', xp, data,
                                            mapping, (3, 8), order=order,
                                            comparator=assert_array_almost_equal)
        assert_array_almost_equal(result[..., ::2], expected[..., ::2])

    def test_geometric_transform20(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)

        def mapping(x):
            return (x[0] / 2, x[1])

        result, expected = compare_to_scipy('geometric_transform', xp, data,
                                            mapping, (6, 4), order=order,
                                            comparator=assert_array_almost_equal)
        assert_array_almost_equal(result[::2, ...], expected[::2, ...])

    def test_geometric_transform21(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)

        def mapping(x):
            return (x[0] / 2, x[1] / 2)

        result, expected = compare_to_scipy('geometric_transform', xp, data,
                                            mapping, (6, 8), order=order,
                                            comparator=assert_array_almost_equal)
        assert_array_almost_equal(result[::2, ::2], expected[::2, ::2])

    def test_geometric_transform22(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)

        def mapping1(x):
            return (x[0] / 2, x[1] / 2)

        def mapping2(x):
            return (x[0] * 2, x[1] * 2)

        intermediate, _ = compare_to_scipy('geometric_transform', xp, data,
                                           mapping1, (6, 8), order=order,
                                           comparator=assert_array_almost_equal)
        compare_to_scipy('geometric_transform', xp, intermediate, mapping2,
                         (3, 4), order=order,
                         comparator=assert_array_almost_equal)

    def test_geometric_transform23(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)

        def mapping(x):
            return (1, x[0] * 2)

        compare_to_scipy('geometric_transform', xp, data, mapping, (2,),
                         order=order, comparator=assert_array_almost_equal)

    def test_geometric_transform24(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)

        def mapping(x, a, b):
            return (a, x[0] * b)

        compare_to_scipy('geometric_transform', xp, data, mapping, (2,),
                         order=order, extra_arguments=(1,),
                         extra_keywords={'b': 2},
                         comparator=assert_array_almost_equal)


@make_xp_test_case(ndimage.geometric_transform)
class TestGeometricTransformExtra:

    def test_geometric_transform_grid_constant_order1(self, xp):

        # verify interpolation outside the original bounds
        x = xp.asarray([[1, 2, 3],
                        [4, 5, 6]], dtype=xp.float64)

        def mapping(x):
            return (x[0] - 0.5), (x[1] - 0.5)

        compare_to_scipy('geometric_transform', xp, x, mapping,
                         mode='grid-constant', order=1,
                         comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('mode', ['grid-constant', 'grid-wrap', 'nearest',
                                      'mirror', 'reflect'])
    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_geometric_transform_vs_padded(self, order, mode, xp):

        def mapping(x):
            return (x[0] - 0.4), (x[1] + 2.3)

        x = xp.asarray(np.arange(144, dtype=float).reshape(12, 12))
        compare_to_scipy('geometric_transform', xp, x, mapping, mode=mode,
                         order=order, comparator=xp_assert_close,
                         comparator_kwargs={'rtol': 1e-7})

    @skip_xp_backends(np_only=True, reason='endianness is numpy-specific')
    def test_geometric_transform_endianness_with_output_parameter(self, xp):
        # geometric transform given output ndarray or dtype with
        # non-native endianness. see issue #4127
        data = np.asarray([1])

        def mapping(x):
            return x

        for out in [data.dtype, data.dtype.newbyteorder(),
                    np.empty_like(data),
                    np.empty_like(data).astype(data.dtype.newbyteorder())]:
            returned = ndimage.geometric_transform(data, mapping, data.shape,
                                                  output=out, order=1)
            result = out if returned is None else returned
            assert_array_almost_equal(result, [1])

    @skip_xp_backends(np_only=True, reason='string `output` is numpy-specific')
    def test_geometric_transform_with_string_output(self, xp):
        data = xp.asarray([1])

        def mapping(x):
            return x

        out = ndimage.geometric_transform(data, mapping, output='f', order=1)
        assert out.dtype is np.dtype('f')
        assert_array_almost_equal(out, [1])


@make_xp_test_case(ndimage.map_coordinates)
class TestMapCoordinates:

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    @pytest.mark.parametrize('dtype', [np.float64, np.complex128])
    def test_map_coordinates01(self, order, dtype, xp):
        if is_jax(xp) and order > 1:
            pytest.xfail("jax map_coordinates requires order <= 1")

        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        if xp.isdtype(data.dtype, 'complex floating'):
            data = data - 1j * data

        idx = np.indices(data.shape)
        idx -= 1
        idx = xp.asarray(idx)

        compare_to_scipy('map_coordinates', xp, data, idx, order=order,
                         comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_map_coordinates02(self, order, xp):
        if is_jax(xp):
            if order > 1:
               pytest.xfail("jax map_coordinates requires order <= 1")
            if order == 1:
               pytest.xfail("output differs. jax bug?")

        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        idx = np.indices(data.shape, np.float64)
        idx -= 0.5
        idx = xp.asarray(idx)

        out1 = ndimage.shift(data, 0.5, order=order)
        out2 = ndimage.map_coordinates(data, idx, order=order)
        assert_array_almost_equal(out1, out2)

    @skip_xp_backends("jax.numpy", reason="`order` is required in jax")
    def test_map_coordinates03(self, xp):
        data = _asarray([[4, 1, 3, 2],
                         [7, 6, 8, 5],
                         [3, 5, 3, 6]], order='F', xp=xp)
        idx = np.indices(data.shape) - 1
        idx = xp.asarray(idx)
        out, expected = compare_to_scipy('map_coordinates', xp, data, idx,
                                         order=1,
                                         comparator=assert_array_almost_equal)
        assert_array_almost_equal(out, ndimage.shift(data, (1, 1), order=1))

        idx = np.indices(data[::2, ...].shape) - 1
        idx = xp.asarray(idx)
        out, expected = compare_to_scipy('map_coordinates', xp,
                                         data[::2, ...], idx, order=1,
                                         comparator=assert_array_almost_equal)
        assert_array_almost_equal(out, ndimage.shift(data[::2, ...], (1, 1), order=1))

        idx = np.indices(data[:, ::2].shape) - 1
        idx = xp.asarray(idx)
        out, expected = compare_to_scipy('map_coordinates', xp,
                                         data[:, ::2], idx, order=1,
                                         comparator=assert_array_almost_equal)
        assert_array_almost_equal(out, ndimage.shift(data[:, ::2], (1, 1), order=1))

    @skip_xp_backends(np_only=True)
    def test_map_coordinates_endianness_with_output_parameter(self, xp):
        # output parameter given as array or dtype with either endianness
        # see issue #4127
        # NB: NumPy-only

        data = np.asarray([[1, 2], [7, 6]])
        expected = np.asarray([[0, 0], [0, 1]])
        idx = np.indices(data.shape)
        idx -= 1
        for out in [
            data.dtype,
            data.dtype.newbyteorder(),
            np.empty_like(expected),
            np.empty_like(expected).astype(expected.dtype.newbyteorder())
        ]:
            returned = ndimage.map_coordinates(data, idx, output=out, order=1)
            result = out if returned is None else returned
            assert_array_almost_equal(result, expected)

    @skip_xp_backends(np_only=True, reason='string `output` is numpy-specific')
    def test_map_coordinates_with_string_output(self, xp):
        data = xp.asarray([[1]])
        idx = np.indices(data.shape)
        idx = xp.asarray(idx)
        out = ndimage.map_coordinates(data, idx, output='f', order=1)
        assert out.dtype is np.dtype('f')
        assert_array_almost_equal(out, xp.asarray([[1]]))

    @pytest.mark.skip_xp_backends(cpu_only=True)
    @pytest.mark.skipif('win32' in sys.platform or np.intp(0).itemsize < 8,
                        reason='do not run on 32 bit or windows '
                               '(no sparse memory)')
    def test_map_coordinates_large_data(self, xp):
        # check crash on large data
        try:
            n = 30000
            # a = xp.reshape(xp.empty(n**2, dtype=xp.float32), (n, n))
            a = np.empty(n**2, dtype=np.float32).reshape(n, n)
            # fill the part we might read
            a[n - 3:, n - 3:] = 0
            ndimage.map_coordinates(
                xp.asarray(a), xp.asarray([[n - 1.5], [n - 1.5]]), order=1
            )
        except MemoryError as e:
            raise pytest.skip('Not enough memory available') from e


@make_xp_test_case(ndimage.affine_transform)
class TestAffineTransform:

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform01(self, order, xp):
        data = xp.asarray([1])
        compare_to_scipy('affine_transform', xp, data, xp.asarray([[1]]),
                         order=order, comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform02(self, order, xp):
        data = xp.ones([4])
        compare_to_scipy('affine_transform', xp, data, xp.asarray([[1]]),
                         order=order, comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform03(self, order, xp):
        data = xp.ones([4])
        compare_to_scipy('affine_transform', xp, data, xp.asarray([[1]]), -1,
                         order=order, comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform04(self, order, xp):
        data = xp.asarray([4, 1, 3, 2])
        compare_to_scipy('affine_transform', xp, data, xp.asarray([[1]]), -1,
                         order=order, comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    @pytest.mark.parametrize('dtype', ["float64", "complex128"])
    def test_affine_transform05(self, order, dtype, xp):
        dtype = getattr(xp, dtype)
        data = xp.asarray([[1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1]], dtype=dtype)
        if xp.isdtype(data.dtype, 'complex floating'):
            data -= 1j * data
        compare_to_scipy('affine_transform', xp, data,
                         xp.asarray([[1, 0], [0, 1]]), [0, -1], order=order,
                         comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform06(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        compare_to_scipy('affine_transform', xp, data,
                         xp.asarray([[1, 0], [0, 1]]), [0, -1], order=order,
                         comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform07(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        compare_to_scipy('affine_transform', xp, data,
                         xp.asarray([[1, 0], [0, 1]]), [-1, 0], order=order,
                         comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform08(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        compare_to_scipy('affine_transform', xp, data,
                         xp.asarray([[1, 0], [0, 1]]), [-1, -1], order=order,
                         comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform09(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        compare_to_scipy('affine_transform', xp, filtered,
                         xp.asarray([[1, 0], [0, 1]]), [-1, -1], order=order,
                         prefilter=False,
                         comparator=assert_array_almost_equal)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform10(self, order, xp):
        data = xp.ones([2], dtype=xp.float64)
        out = ndimage.affine_transform(data, xp.asarray([[0.5]]), output_shape=(4,),
                                       order=order)
        assert_array_almost_equal(out, xp.asarray([1, 1, 1, 0]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform11(self, order, xp):
        data = xp.asarray([1, 5, 2, 6, 3, 7, 4, 4])
        out = ndimage.affine_transform(data, xp.asarray([[2]]), 0, (4,), order=order)
        assert_array_almost_equal(out, xp.asarray([1, 2, 3, 4]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform12(self, order, xp):
        data = xp.asarray([1, 2, 3, 4])
        out = ndimage.affine_transform(data, xp.asarray([[0.5]]), 0, (8,), order=order)
        assert_array_almost_equal(out[::2], xp.asarray([1, 2, 3, 4]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform13(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9.0, 10, 11, 12]]
        data = xp.asarray(data)
        out = ndimage.affine_transform(data, xp.asarray([[1, 0], [0, 2]]), 0, (3, 2),
                                       order=order)
        assert_array_almost_equal(out, xp.asarray([[1, 3], [5, 7], [9, 11]]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform14(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)
        out = ndimage.affine_transform(data, xp.asarray([[2, 0], [0, 1]]), 0, (1, 4),
                                       order=order)
        assert_array_almost_equal(out, xp.asarray([[1, 2, 3, 4]]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform15(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)
        out = ndimage.affine_transform(data, xp.asarray([[2, 0], [0, 2]]), 0, (1, 2),
                                       order=order)
        assert_array_almost_equal(out, xp.asarray([[1, 3]]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform16(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)
        out = ndimage.affine_transform(data, xp.asarray([[1, 0.0], [0, 0.5]]), 0,
                                       (3, 8), order=order)
        assert_array_almost_equal(out[..., ::2], data)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform17(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)
        out = ndimage.affine_transform(data, xp.asarray([[0.5, 0], [0, 1]]), 0,
                                       (6, 4), order=order)
        assert_array_almost_equal(out[::2, ...], data)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform18(self, order, xp):
        data = xp.asarray([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]])
        out = ndimage.affine_transform(data, xp.asarray([[0.5, 0], [0, 0.5]]), 0,
                                       (6, 8), order=order)
        assert_array_almost_equal(out[::2, ::2], data)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform19(self, order, xp):
        data = xp.asarray([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]], dtype=xp.float64)
        out = ndimage.affine_transform(data, xp.asarray([[0.5, 0], [0, 0.5]]), 0,
                                       (6, 8), order=order)
        out = ndimage.affine_transform(out, xp.asarray([[2.0, 0], [0, 2.0]]), 0,
                                       (3, 4), order=order)
        assert_array_almost_equal(out, data)

    @xfail_xp_backends("cupy", reason="https://github.com/cupy/cupy/issues/8394")
    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform20(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)
        out = ndimage.affine_transform(data, xp.asarray([[0], [2]]), 0, (2,),
                                       order=order)
        assert_array_almost_equal(out, xp.asarray([1, 3]))

    @xfail_xp_backends("cupy", reason="https://github.com/cupy/cupy/issues/8394")
    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform21(self, order, xp):
        data = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]
        data = xp.asarray(data)
        out = ndimage.affine_transform(data, xp.asarray([[2], [0]]), 0, (2,),
                                       order=order)
        assert_array_almost_equal(out, xp.asarray([1, 9]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform22(self, order, xp):
        # shift and offset interaction; see issue #1547
        data = xp.asarray([4, 1, 3, 2])
        out = ndimage.affine_transform(data, xp.asarray([[2]]), [-1], (3,),
                                       order=order)
        assert_array_almost_equal(out, xp.asarray([0, 1, 2]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform23(self, order, xp):
        # shift and offset interaction; see issue #1547
        data = xp.asarray([4, 1, 3, 2])
        out = ndimage.affine_transform(data, xp.asarray([[0.5]]), [-1], (8,),
                                       order=order)
        assert_array_almost_equal(out[::2], xp.asarray([0, 4, 1, 3]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform24(self, order, xp):
        # consistency between diagonal and non-diagonal case; see issue #1547
        data = xp.asarray([4, 1, 3, 2])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                'The behavior of affine_transform with a 1-D array .* has changed',
                UserWarning)
            out1 = ndimage.affine_transform(data, xp.asarray([2]), -1, order=order)
        out2 = ndimage.affine_transform(data, xp.asarray([[2]]), -1, order=order)
        assert_array_almost_equal(out1, out2)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform25(self, order, xp):
        # consistency between diagonal and non-diagonal case; see issue #1547
        data = xp.asarray([4, 1, 3, 2])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                       'The behavior of affine_transform with a 1-D array .* '
                       'has changed', UserWarning)
            out1 = ndimage.affine_transform(data, xp.asarray([0.5]), -1, order=order)
        out2 = ndimage.affine_transform(data, xp.asarray([[0.5]]), -1, order=order)
        assert_array_almost_equal(out1, out2)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform26(self, order, xp):
        # test homogeneous coordinates
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        tform_original = xp.eye(2)
        offset_original = -xp.ones((2, 1))

        tform_h1 = xp.concat((tform_original, offset_original), axis=1)  # hstack
        tform_h2 = xp.concat((tform_h1, xp.asarray([[0.0, 0, 1]])), axis=0)  # vstack

        offs = [float(x) for x in xp.reshape(offset_original, (-1,))]

        out1 = ndimage.affine_transform(filtered, tform_original,
                                        offs,
                                        order=order, prefilter=False)
        out2 = ndimage.affine_transform(filtered, tform_h1, order=order,
                                        prefilter=False)
        out3 = ndimage.affine_transform(filtered, tform_h2, order=order,
                                        prefilter=False)
        for out in [out1, out2, out3]:
            assert_array_almost_equal(out, xp.asarray([[0, 0, 0, 0],
                                                       [0, 4, 1, 3],
                                                       [0, 7, 6, 8]]))

    @xfail_xp_backends("cupy", reason="does not raise")
    def test_affine_transform27(self, xp):
        # test valid homogeneous transformation matrix
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        tform_h1 = xp.concat((xp.eye(2), -xp.ones((2, 1))) , axis=1)  # vstack
        tform_h2 = xp.concat((tform_h1, xp.asarray([[5.0, 2, 1]])), axis=0)  # hstack

        assert_raises(ValueError, ndimage.affine_transform, data, tform_h2, order=1)

    @skip_xp_backends(np_only=True, reason='byteorder is numpy-specific')
    def test_affine_transform_1d_endianness_with_output_parameter(self, xp):
        # 1d affine transform given output ndarray or dtype with
        # either endianness. see issue #7388
        data = xp.ones((2, 2))
        for out in [xp.empty_like(data),
                    xp.empty_like(data).astype(data.dtype.newbyteorder()),
                    data.dtype, data.dtype.newbyteorder()]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                           'The behavior of affine_transform with a 1-D array '
                           '.* has changed', UserWarning)
                matrix = xp.asarray([1, 1])
                returned = ndimage.affine_transform(data, matrix, output=out, order=1)
            result = out if returned is None else returned
            assert_array_almost_equal(result, xp.asarray([[1, 1], [1, 1]]))

    @skip_xp_backends(np_only=True, reason='byteorder is numpy-specific')
    def test_affine_transform_multi_d_endianness_with_output_parameter(self, xp):
        # affine transform given output ndarray or dtype with either endianness
        # see issue #4127
        # NB: byteorder is numpy-specific
        data = np.asarray([1])
        for out in [data.dtype, data.dtype.newbyteorder(),
                    np.empty_like(data),
                    np.empty_like(data).astype(data.dtype.newbyteorder())]:
            returned = ndimage.affine_transform(data, np.asarray([[1]]), output=out, order=1)
            result = out if returned is None else returned
            assert_array_almost_equal(result, np.asarray([1]))

    @skip_xp_backends(np_only=True,
        reason='`out` of a different size is numpy-specific'
    )
    def test_affine_transform_output_shape(self, xp):
        # don't require output_shape when out of a different size is given
        data = xp.arange(8, dtype=xp.float64)
        out = xp.ones((16,))

        ndimage.affine_transform(data, xp.asarray([[1]]), output=out, order=1)
        assert_array_almost_equal(out[:8], data)

        # mismatched output shape raises an error
        with pytest.raises(RuntimeError):
            ndimage.affine_transform(
                data, [[1]], output=out, output_shape=(12,))

    @skip_xp_backends(np_only=True, reason='string `output` is numpy-specific')
    def test_affine_transform_with_string_output(self, xp):
        data = xp.asarray([1])
        out = ndimage.affine_transform(data, xp.asarray([[1]]), output='f', order=1)
        assert out.dtype is np.dtype('f')
        assert_array_almost_equal(out, xp.asarray([1]))

    @pytest.mark.parametrize('shift',
                             [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform_shift_via_grid_wrap(self, shift, order, xp):
        # For mode 'grid-wrap', integer shifts should match np.roll
        x = np.asarray([[0, 1],
                        [2, 3]])
        affine = np.zeros((2, 3))
        affine[:2, :2] = np.eye(2)
        affine[:, 2] = np.asarray(shift)

        expected = np.roll(x, shift, axis=(0, 1))

        x = xp.asarray(x)
        affine = xp.asarray(affine)
        expected = xp.asarray(expected)

        assert_array_almost_equal(
            ndimage.affine_transform(x, affine, mode='grid-wrap', order=order),
            expected
        )

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_affine_transform_shift_reflect(self, order, xp):
        # shift by x.shape results in reflection
        x = np.asarray([[0, 1, 2],
                        [3, 4, 5]])
        expected = x[::-1, ::-1].copy()   # strides >0 for torch
        x = xp.asarray(x)
        expected = xp.asarray(expected)

        affine = np.zeros([2, 3])
        affine[:2, :2] = np.eye(2)
        affine[:, 2] = np.asarray(x.shape)
        affine = xp.asarray(affine)

        assert_array_almost_equal(
            ndimage.affine_transform(x, affine, mode='reflect', order=order),
            expected,
        )


@make_xp_test_case(ndimage.shift)
class TestShift:

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift01(self, order, xp):
        data = xp.asarray([1])
        out = ndimage.shift(data, [1], order=order)
        assert_array_almost_equal(out, xp.asarray([0]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift02(self, order, xp):
        data = xp.ones([4])
        out = ndimage.shift(data, [1], order=order)
        assert_array_almost_equal(out, xp.asarray([0, 1, 1, 1]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift03(self, order, xp):
        data = xp.ones([4])
        out = ndimage.shift(data, -1, order=order)
        assert_array_almost_equal(out, xp.asarray([1, 1, 1, 0]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift04(self, order, xp):
        data = xp.asarray([4, 1, 3, 2])
        out = ndimage.shift(data, 1, order=order)
        assert_array_almost_equal(out, xp.asarray([0, 4, 1, 3]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    @pytest.mark.parametrize('dtype', ["float64", "complex128"])
    def test_shift05(self, order, dtype, xp):
        dtype = getattr(xp, dtype)
        data = xp.asarray([[1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1]], dtype=dtype)
        expected = xp.asarray([[0, 1, 1, 1],
                               [0, 1, 1, 1],
                               [0, 1, 1, 1]], dtype=dtype)
        if xp.isdtype(data.dtype, 'complex floating'):
            data -= 1j * data
            expected -= 1j * expected
        out = ndimage.shift(data, [0, 1], order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    @pytest.mark.parametrize('mode', ['constant', 'grid-constant'])
    @pytest.mark.parametrize('dtype', ['float64', 'complex128'])
    def test_shift_with_nonzero_cval(self, order, mode, dtype, xp):
        data = np.asarray([[1, 1, 1, 1],
                           [1, 1, 1, 1],
                           [1, 1, 1, 1]], dtype=dtype)

        expected = np.asarray([[0, 1, 1, 1],
                               [0, 1, 1, 1],
                               [0, 1, 1, 1]], dtype=dtype)

        if np_compat.isdtype(data.dtype, 'complex floating'):
            data -= 1j * data
            expected -= 1j * expected
        cval = 5.0
        expected[:, 0] = cval  # specific to shift of [0, 1] used below

        data = xp.asarray(data)
        expected = xp.asarray(expected)
        out = ndimage.shift(data, [0, 1], order=order, mode=mode, cval=cval)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift06(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        out = ndimage.shift(data, [0, 1], order=order)
        assert_array_almost_equal(out, xp.asarray([[0, 4, 1, 3],
                                                   [0, 7, 6, 8],
                                                   [0, 3, 5, 3]]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift07(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        out = ndimage.shift(data, [1, 0], order=order)
        assert_array_almost_equal(out, xp.asarray([[0, 0, 0, 0],
                                                   [4, 1, 3, 2],
                                                   [7, 6, 8, 5]]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift08(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        out = ndimage.shift(data, [1, 1], order=order)
        assert_array_almost_equal(out, xp.asarray([[0, 0, 0, 0],
                                                   [0, 4, 1, 3],
                                                   [0, 7, 6, 8]]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift09(self, order, xp):
        data = xp.asarray([[4, 1, 3, 2],
                           [7, 6, 8, 5],
                           [3, 5, 3, 6]])
        if (order > 1):
            filtered = ndimage.spline_filter(data, order=order)
        else:
            filtered = data
        out = ndimage.shift(filtered, [1, 1], order=order, prefilter=False)
        assert_array_almost_equal(out, xp.asarray([[0, 0, 0, 0],
                                                   [0, 4, 1, 3],
                                                   [0, 7, 6, 8]]))

    @pytest.mark.parametrize('shift',
                             [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift_grid_wrap(self, shift, order, xp):
        # For mode 'grid-wrap', integer shifts should match np.roll
        x = np.asarray([[0, 1],
                        [2, 3]])
        expected = np.roll(x, shift, axis=(0,1))

        x = xp.asarray(x)
        expected = xp.asarray(expected)

        assert_array_almost_equal(
            ndimage.shift(x, shift, mode='grid-wrap', order=order),
            expected
        )

    @pytest.mark.parametrize('shift',
                             [(1, 0), (0, 1), (-1, 1), (3, -5), (2, 7)])
    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift_grid_constant1(self, shift, order, xp):
        # For integer shifts, 'constant' and 'grid-constant' should be equal
        x = xp.reshape(xp.arange(20), (5, 4))
        assert_array_almost_equal(
            ndimage.shift(x, shift, mode='grid-constant', order=order),
            ndimage.shift(x, shift, mode='constant', order=order),
        )

    def test_shift_grid_constant_order1(self, xp):
        x = xp.asarray([[1, 2, 3],
                        [4, 5, 6]], dtype=xp.float64)
        expected_result = xp.asarray([[0.25, 0.75, 1.25],
                                      [1.25, 3.00, 4.00]])
        assert_array_almost_equal(
            ndimage.shift(x, (0.5, 0.5), mode='grid-constant', order=1),
            expected_result,
        )

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift_reflect(self, order, xp):
        # shift by x.shape results in reflection
        x = np.asarray([[0, 1, 2],
                        [3, 4, 5]])
        expected = x[::-1, ::-1].copy()   # strides > 0 for torch

        x = xp.asarray(x)
        expected = xp.asarray(expected)
        assert_array_almost_equal(
            ndimage.shift(x, x.shape, mode='reflect', order=order),
            expected,
        )

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    @pytest.mark.parametrize('prefilter', [False, True])
    def test_shift_nearest_boundary(self, order, prefilter, xp):
        # verify that shifting at least order // 2 beyond the end of the array
        # gives a value equal to the edge value.
        x = xp.arange(16)
        kwargs = dict(mode='nearest', order=order, prefilter=prefilter)
        assert_array_almost_equal(
            ndimage.shift(x, order // 2 + 1, **kwargs)[0], x[0],
        )
        assert_array_almost_equal(
            ndimage.shift(x, -order // 2 - 1, **kwargs)[-1], x[-1],
        )

    @pytest.mark.parametrize('mode', ['grid-constant', 'grid-wrap', 'nearest',
                                      'mirror', 'reflect'])
    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_shift_vs_padded(self, order, mode, xp):
        x_np = np.arange(144, dtype=float).reshape(12, 12)
        shift = (0.4, -2.3)

        # manually pad and then extract center to get expected result
        npad = 32
        pad_mode = ndimage_to_numpy_mode.get(mode)
        x_padded = xp.asarray(np.pad(x_np, npad, mode=pad_mode))
        x = xp.asarray(x_np)

        center_slice = tuple([slice(npad, -npad)] * x.ndim)
        expected_result = ndimage.shift(
            x_padded, shift, mode=mode, order=order)[center_slice]

        xp_assert_close(
            ndimage.shift(x, shift, mode=mode, order=order),
            expected_result,
            rtol=1e-7,
        )


@make_xp_test_case(ndimage.zoom)
class TestZoom:

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_zoom1(self, order, xp):
        for z in [2, [2, 2]]:
            arr = xp.reshape(xp.arange(25, dtype=xp.float64), (5, 5))
            arr = ndimage.zoom(arr, z, order=order)
            assert arr.shape == (10, 10)
            assert xp.all(arr[-1, :] != 0)
            assert xp.all(arr[-1, :] >= (20 - eps))
            assert xp.all(arr[0, :] <= (5 + eps))
            assert xp.all(arr >= (0 - eps))
            assert xp.all(arr <= (24 + eps))

    def test_zoom2(self, xp):
        arr = xp.reshape(xp.arange(12), (3, 4))
        out = ndimage.zoom(ndimage.zoom(arr, 2, order=1), 0.5, order=1)
        xp_assert_equal(out, arr)

    def test_zoom3(self, xp):
        arr = xp.asarray([[1, 2]])
        out1 = ndimage.zoom(arr, (2, 1), order=1)
        out2 = ndimage.zoom(arr, (1, 2), order=1)

        assert_array_almost_equal(out1, xp.asarray([[1, 2], [1, 2]]))
        assert_array_almost_equal(out2, xp.asarray([[1, 1, 2, 2]]))

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    @pytest.mark.parametrize('dtype', ["float64", "complex128"])
    def test_zoom_affine01(self, order, dtype, xp):
        dtype = getattr(xp, dtype)
        data = xp.asarray([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]], dtype=dtype)
        if xp.isdtype(data.dtype, 'complex floating'):
            data -= 1j * data
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                       'The behavior of affine_transform with a 1-D array .* '
                       'has changed', UserWarning)
            out = ndimage.affine_transform(data, xp.asarray([0.5, 0.5]), 0,
                                           (6, 8), order=order)
        assert_array_almost_equal(out[::2, ::2], data)

    def test_zoom_infinity(self, xp):
        # Ticket #1419 regression test
        dim = 8
        ndimage.zoom(xp.zeros((dim, dim)), 1. / dim, mode='nearest', order=1)

    def test_zoom_zoomfactor_one(self, xp):
        # Ticket #1122 regression test
        arr = xp.zeros((1, 5, 5))
        zoom = (1.0, 2.0, 2.0)

        out = ndimage.zoom(arr, zoom, cval=7, order=1)
        ref = xp.zeros((1, 10, 10))
        assert_array_almost_equal(out, ref)

    def test_zoom_output_shape_roundoff(self, xp):
        arr = xp.zeros((3, 11, 25))
        zoom = (4.0 / 3, 15.0 / 11, 29.0 / 25)
        out = ndimage.zoom(arr, zoom, order=1)
        assert out.shape == (4, 15, 29)

    @pytest.mark.parametrize('zoom', [(1, 1), (3, 5), (8, 2), (8, 8)])
    @pytest.mark.parametrize('mode', ['nearest', 'constant', 'wrap', 'reflect',
                                      'mirror', 'grid-wrap', 'grid-mirror',
                                      'grid-constant'])
    def test_zoom_by_int_order0(self, zoom, mode, xp):
        # order 0 zoom should be the same as replication via np.kron
        # Note: This is not True for general x shapes when grid_mode is False,
        #       but works here for all modes because the size ratio happens to
        #       always be an integer when x.shape = (2, 2).
        x_np = np.asarray([[0, 1],
                           [2, 3]], dtype=np.float64)
        expected = np.kron(x_np, np.ones(zoom))

        x = xp.asarray(x_np)
        expected = xp.asarray(expected)

        assert_array_almost_equal(
            ndimage.zoom(x, zoom, order=0, mode=mode),
            expected
        )

    @pytest.mark.parametrize('shape', [(2, 3), (4, 4)])
    @pytest.mark.parametrize('zoom', [(1, 1), (3, 5), (8, 2), (8, 8)])
    @pytest.mark.parametrize('mode', ['nearest', 'reflect', 'mirror',
                                      'grid-wrap', 'grid-constant'])
    def test_zoom_grid_by_int_order0(self, shape, zoom, mode, xp):
        # When grid_mode is True,  order 0 zoom should be the same as
        # replication via np.kron. The only exceptions to this are the
        # non-grid modes 'constant' and 'wrap'.
        x_np = np.arange(np.prod(shape), dtype=float).reshape(shape)

        x = xp.asarray(x_np)
        assert_array_almost_equal(
            ndimage.zoom(x, zoom, order=0, mode=mode, grid_mode=True),
            xp.asarray(np.kron(x_np, np.ones(zoom)))
        )

    @pytest.mark.parametrize('mode', ['constant', 'wrap'])
    def test_zoom_grid_mode_warnings(self, mode, xp):
        # Warn on use of non-grid modes when grid_mode is True
        x = xp.reshape(xp.arange(9, dtype=xp.float64), (3, 3))
        with pytest.warns(UserWarning,
                          match="It is recommended to use mode"):
            ndimage.zoom(x, 2, mode=mode, grid_mode=True, order=1),

    @skip_xp_backends("dask.array", reason="output=array requires buffer view")
    @skip_xp_backends("jax.numpy", reason="output=array requires buffer view")
    def test_zoom_output_shape(self, xp):
        """Ticket #643"""
        x = xp.reshape(xp.arange(12), (3, 4))
        ndimage.zoom(x, 2, output=xp.zeros((6, 8)), order=1)

    def test_zoom_0d_array(self, xp):
        # Ticket #21670 regression test
        a = xp.arange(10.)
        factor = 2
        actual = ndimage.zoom(a, np.array(factor), order=1)
        expected = ndimage.zoom(a, factor, order=1)
        xp_assert_close(actual, expected)

    @xfail_xp_backends("cupy", reason="CuPy `zoom` needs similar fix.")
    def test_zoom_1_gh20999(self, xp):
        # gh-20999 reported that zoom with `zoom=1` (or sequence of ones)
        # introduced noise. Check that this is resolved.
        x = xp.eye(3)
        xp_assert_equal(ndimage.zoom(x, 1, order=1), x)
        xp_assert_equal(ndimage.zoom(x, (1, 1), order=1), x)

    @xfail_xp_backends("cupy", reason="CuPy `zoom` needs similar fix.")
    @skip_xp_backends("jax.numpy", reason="read-only backend")
    @xfail_xp_backends("dask.array", reason="numpy round-trip")
    def test_zoom_1_gh20999_output(self, xp):
        x = xp.eye(3)
        output = xp.zeros_like(x)
        ndimage.zoom(x, 1, output=output, order=1)
        xp_assert_equal(output, x)


@make_xp_test_case(ndimage.rotate)
class TestRotate:

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_rotate01(self, order, xp):
        data = xp.asarray([[0, 0, 0, 0],
                           [0, 1, 1, 0],
                           [0, 0, 0, 0]], dtype=xp.float64)
        out = ndimage.rotate(data, 0, order=order)
        assert_array_almost_equal(out, data)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_rotate02(self, order, xp):
        data = xp.asarray([[0, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0]], dtype=xp.float64)
        expected = xp.asarray([[0, 0, 0],
                               [0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]], dtype=xp.float64)
        out = ndimage.rotate(data, 90, order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    @pytest.mark.parametrize('dtype', ["float64", "complex128"])
    def test_rotate03(self, order, dtype, xp):
        dtype = getattr(xp, dtype)
        data = xp.asarray([[0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0]], dtype=dtype)
        expected = xp.asarray([[0, 0, 0],
                               [0, 0, 0],
                               [0, 1, 0],
                               [0, 1, 0],
                               [0, 0, 0]], dtype=dtype)
        if xp.isdtype(data.dtype, 'complex floating'):
            data -= 1j * data
            expected -= 1j * expected
        out = ndimage.rotate(data, 90, order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_rotate04(self, order, xp):
        data = xp.asarray([[0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0]], dtype=xp.float64)
        expected = xp.asarray([[0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0]], dtype=xp.float64)
        out = ndimage.rotate(data, 90, reshape=False, order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_rotate05(self, order, xp):
        data = np.empty((4, 3, 3))
        for i in range(3):
            data[:, :, i] = np.asarray([[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]], dtype=np.float64)
        data = xp.asarray(data)
        expected = xp.asarray([[0, 0, 0, 0],
                               [0, 1, 1, 0],
                               [0, 0, 0, 0]], dtype=xp.float64)
        out = ndimage.rotate(data, 90, order=order)
        for i in range(3):
            assert_array_almost_equal(out[:, :, i], expected)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_rotate06(self, order, xp):
        data = np.empty((3, 4, 3))
        for i in range(3):
            data[:, :, i] = np.asarray([[0, 0, 0, 0],
                                        [0, 1, 1, 0],
                                        [0, 0, 0, 0]], dtype=np.float64)
        data = xp.asarray(data)
        expected = xp.asarray([[0, 0, 0],
                               [0, 1, 0],
                               [0, 1, 0],
                               [0, 0, 0]], dtype=xp.float64)
        out = ndimage.rotate(data, 90, order=order)
        for i in range(3):
            assert_array_almost_equal(out[:, :, i], expected)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_rotate07(self, order, xp):
        data = xp.asarray([[[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]]] * 2, dtype=xp.float64)
        data = xp.permute_dims(data, (2, 1, 0))
        expected = xp.asarray([[[0, 0, 0],
                                [0, 1, 0],
                                [0, 1, 0],
                                [0, 0, 0],
                                [0, 0, 0]]] * 2, dtype=xp.float64)
        expected = xp.permute_dims(expected, (2, 1, 0))
        out = ndimage.rotate(data, 90, axes=(0, 1), order=order)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('order', SUPPORTED_ORDERS)
    def test_rotate08(self, order, xp):
        data = xp.asarray([[[0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0]]] * 2, dtype=xp.float64)
        data = xp.permute_dims(data, (2, 1, 0))  # == np.transpose
        expected = xp.asarray([[[0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 0, 0]]] * 2, dtype=xp.float64)
        expected = xp.permute_dims(expected, (2, 1, 0))
        out = ndimage.rotate(data, 90, axes=(0, 1), reshape=False, order=order)
        assert_array_almost_equal(out, expected)

    def test_rotate09(self, xp):
        data = xp.asarray([[0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0]] * 2, dtype=xp.float64)
        with assert_raises(ValueError):
            ndimage.rotate(data, 90, axes=(0, data.ndim), order=1)

    def test_rotate10(self, xp):
        data = xp.reshape(xp.arange(45, dtype=xp.float64), (3, 5, 3))

	# The output of ndimage.rotate before refactoring
        expected = xp.asarray([[[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [6.95152106, 7.95152106, 8.95152106],
                                [13.00463923, 14.00463923, 15.00463923],
                                [0.0, 0.0, 0.0]],
                               [[8.89376367, 9.89376367, 10.89376367],
                                [14.94688184, 15.94688184, 16.94688184],
                                [21.0, 22.0, 23.0],
                                [27.05311816, 28.05311816, 29.05311816],
                                [33.10623633, 34.10623633, 35.10623633]],
                               [[0.0, 0.0, 0.0],
                                [28.99536077, 29.99536077, 30.99536077],
                                [35.04847894, 36.04847894, 37.04847894],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]]], dtype=xp.float64)

        out = ndimage.rotate(data, angle=12, reshape=False, order=1)
        #assert_array_almost_equal(out, expected)
        xp_assert_close(out, expected, rtol=1e-6, atol=2e-6)


    @xfail_xp_backends("cupy", reason="https://github.com/cupy/cupy/issues/8400")
    def test_rotate_exact_180(self, xp):
        a = xp.asarray(np.tile(np.arange(5), (5, 1)))
        b = ndimage.rotate(ndimage.rotate(a, 180, order=1), -180, order=1)
        xp_assert_equal(a, b)
