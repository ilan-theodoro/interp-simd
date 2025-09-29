# interp-ndimage

`interp-ndimage` is a C++/pybind11 port of SciPy's `scipy.ndimage` interpolation routines (orders 0 and 1 only). The module mirrors SciPy's Python API and behaviour while reusing the original vectorized interpolation kernels.

## Features

- Drop-in replacement for `scipy.ndimage` interpolation helpers: `affine_transform`, `geometric_transform`, `map_coordinates`, `shift`, `zoom`, and related entry points.
- Supports the same dtype matrix as SciPy, including complex floating point types and little/big endian variations.
- Runtime dispatch between AVX2-accelerated kernels and scalar fallbacks, matching SciPy's source implementation.
- Parallel execution via OpenMP, with thread-safe kernels and deterministic behaviour.

## Building

```
pip install .
```

During the build, the extension checks compiler support for AVX2/F16C and OpenMP. You can disable OpenMP by setting `INTERP_NDIMAGE_NO_OPENMP=1` in the environment.

## Testing

```
pytest
```

## Licensing

This project bundles and adapts code from SciPy (BSD-3-Clause); see `LICENSE` for details.
