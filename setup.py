from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

try:
    import numpy
except ImportError as exc:  # pragma: no cover - handled by build requirements
    raise SystemExit("numpy must be installed before building interp-ndimage") from exc


def has_flag(compiler, flag: str) -> bool:
    """Return whether the given flag is supported by the compiler."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write("int main() { return 0; }\n")
        fname = Path(f.name)

    try:
        compiler.compile([str(fname)], extra_postargs=[flag])
    except Exception:  # pragma: no cover - depends on compiler
        return False
    finally:
        try:
            fname.unlink()
        except OSError:
            pass
    return True


def get_openmp_flags(compiler):
    if os.environ.get("INTERP_NDIMAGE_NO_OPENMP") == "1":
        return [], []
    cflags, lflags = [], []
    ctype = compiler.compiler_type
    if ctype == "msvc":
        if has_flag(compiler, "/openmp"):
            cflags.append("/openmp")
    else:
        if has_flag(compiler, "-fopenmp"):
            cflags.append("-fopenmp")
            lflags.append("-fopenmp")
    return cflags, lflags


def get_simd_flags(compiler):
    cflags = []
    if compiler.compiler_type == "msvc":
        if has_flag(compiler, "/arch:AVX2"):
            cflags.extend(["/arch:AVX2"])
    else:
        for flag in ("-mavx2", "-mfma", "-mf16c"):
            if has_flag(compiler, flag):
                cflags.append(flag)
    return cflags


class BuildExt(build_ext):
    def build_extensions(self):
        compiler = self.compiler
        openmp_cflags, openmp_lflags = get_openmp_flags(compiler)
        simd_flags = get_simd_flags(compiler)

        for ext in self.extensions:
            ext.include_dirs.extend([numpy.get_include(), "include", "cpp"])
            ext.define_macros.append(("PYBIND11_DETAILED_ERROR_MESSAGES", "1"))
            ext.extra_compile_args.extend(["-O3", "-fvisibility=hidden", "-Wall", "-Wextra", "-Wpedantic"])
            ext.extra_link_args.extend([])

            # Apply compiler-specific flags after defaults to avoid duplication
            ext.extra_compile_args.extend(openmp_cflags)
            ext.extra_link_args.extend(openmp_lflags)
            ext.extra_compile_args.extend(simd_flags)

        super().build_extensions()


ext_modules = [
    Pybind11Extension(
        "interp_ndimage._nd_image",
        sources=[
            "src/interp_ndimage/_nd_image.cpp",
            "cpp/nd_image_interpolation.c",
            "cpp/ni_interpolation.c",
            "cpp/ni_interpolation_avx2.c",
            "cpp/ni_splines.c",
            "cpp/ni_support.c",
        ],
        define_macros=[("PY_ARRAY_UNIQUE_SYMBOL", "INTERP_NDIMAGE_ARRAY_API")],
        extra_compile_args=[],
        extra_link_args=[],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
