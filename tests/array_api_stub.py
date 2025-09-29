from __future__ import annotations

import functools
from typing import Any, Callable

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as np_assert_array_almost_equal
from numpy.testing import assert_allclose as np_assert_allclose

ArrayLike = Any


def _asarray(array: ArrayLike, dtype: Any = None, order: str | None = None, **kwargs: Any) -> np.ndarray:
    return np.asarray(array, dtype=dtype, order=order)


def assert_array_almost_equal(actual: ArrayLike, desired: ArrayLike) -> None:
    np_assert_array_almost_equal(actual, desired)


def is_jax(xp: Any) -> bool:  # pragma: no cover - trivial stub
    return False


np_compat = np


def xp_assert_equal(actual: np.ndarray, desired: np.ndarray, **kwargs: Any) -> None:
    np.testing.assert_equal(actual, desired)


def xp_assert_close(actual: np.ndarray, desired: np.ndarray, rtol: float = 1e-7, atol: float = 0.0, **kwargs: Any) -> None:
    np_assert_allclose(actual, desired, rtol=rtol, atol=atol)


def make_xp_test_case(*funcs: Callable[..., Any], **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    param = pytest.mark.parametrize("xp", [np])

    def decorator(test_func: Callable[..., Any]) -> Callable[..., Any]:
        return param(test_func)

    return decorator
