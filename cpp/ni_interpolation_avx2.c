/* AVX2 optimized geometric transform implementations for scipy.ndimage
 * Handles 2D bilinear and 3D trilinear interpolation for float32, float64, and
 * float16
 */

#define USE_PREFETCH 0 /* Toggle prefetching for testing: 0=off, 1=on */

#include <omp.h>
#include "ni_support.h"
#include "ni_interpolation.h"
#include <immintrin.h>
#include <math.h>

/* Alignment macros */
#if defined(__GNUC__) || defined(__clang__)
#define SCIPY_ALIGN(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#define SCIPY_ALIGN(x) __declspec(align(x))
#else
#define SCIPY_ALIGN(x)
#endif

#ifdef __AVX2__

/* AVX2 optimized 2D bilinear interpolation for float64 - interior only */
int NI_GeometricTransform_2D_bilinear_f64_avx2(PyArrayObject *input,
                                               PyArrayObject *output,
                                               const double *M,
                                               const double *shift, int nprepad,
                                               int mode, double cval) {
  const npy_intp H = PyArray_DIM(input, 0);
  const npy_intp W = PyArray_DIM(input, 1);
  const npy_intp OH = PyArray_DIM(output, 0);
  const npy_intp OW = PyArray_DIM(output, 1);
  const double *in_data = (const double *)PyArray_DATA(input);
  double *out_data = (double *)PyArray_DATA(output);
  const npy_intp in_stride_y = PyArray_STRIDE(input, 0) / sizeof(double);
  const npy_intp in_stride_x = PyArray_STRIDE(input, 1) / sizeof(double);
  const npy_intp out_stride_y = PyArray_STRIDE(output, 0) / sizeof(double);
  const npy_intp out_stride_x = PyArray_STRIDE(output, 1) / sizeof(double);

  /* Reject tiny images and check for potential overflow */
  if (H < 2 || W < 2)
    return 0;
  if (H > INT_MAX - 1 || W > INT_MAX - 1)
    return 0; /* Prevent int overflow */

  /* Matrix and shift as doubles - SciPy axis convention */
  const double m00 = M[0], m01 = M[1];   /* contributes to y_src (axis 0) */
  const double m10 = M[2], m11 = M[3];   /* contributes to x_src (axis 1) */
  const double shy = shift[0] + nprepad; /* shift for rows (y) */
  const double shx = shift[1] + nprepad; /* shift for cols (x) */

  const __m256d one = _mm256_set1_pd(1.0);

/* Process output pixels */
#pragma omp parallel for schedule(static)
  for (npy_intp oy = 0; oy < OH; ++oy) {
    double *out_row = out_data + oy * out_stride_y;

    /* SciPy: y_src uses row 0 of M + shift[0]; x_src uses row 1 + shift[1] */
    const double base_y = m00 * (double)oy + shy; /* y_src base */
    const double base_x = m10 * (double)oy + shx; /* x_src base */

    npy_intp ox = 0;

    /* AVX2 loop - process 4 pixels at a time */
    for (; ox + 3 < OW; ox += 4) {
      /* Generate x indices [ox, ox+1, ox+2, ox+3] */
      __m256d vx = _mm256_setr_pd((double)ox, (double)(ox + 1),
                                  (double)(ox + 2), (double)(ox + 3));

      /* Compute source coordinates using SciPy axis convention */
      __m256d ys = _mm256_fmadd_pd(_mm256_set1_pd(m01), vx,
                                   _mm256_set1_pd(base_y)); /* y_src */
      __m256d xs = _mm256_fmadd_pd(_mm256_set1_pd(m11), vx,
                                   _mm256_set1_pd(base_x)); /* x_src */

      /* Floor to get integer coordinates */
      __m256d x_floor = _mm256_floor_pd(xs);
      __m256d y_floor = _mm256_floor_pd(ys);

      /* Get fractional parts */
      __m256d fx = _mm256_sub_pd(xs, x_floor);
      __m256d fy = _mm256_sub_pd(ys, y_floor);

      /* Compute bilinear weights */
      __m256d one_minus_fx = _mm256_sub_pd(one, fx);
      __m256d one_minus_fy = _mm256_sub_pd(one, fy);

      __m256d w00 = _mm256_mul_pd(one_minus_fx, one_minus_fy);
      __m256d w01 = _mm256_mul_pd(fx, one_minus_fy);
      __m256d w10 = _mm256_mul_pd(one_minus_fx, fy);
      __m256d w11 = _mm256_mul_pd(fx, fy);

      /* Convert floor values to integers for indexing */
      __m128i xi = _mm256_cvttpd_epi32(x_floor);
      __m128i yi = _mm256_cvttpd_epi32(y_floor);

      /* Extract integer coordinates and weights */
      SCIPY_ALIGN(32) int xi_arr[4];
      SCIPY_ALIGN(32) int yi_arr[4];
      SCIPY_ALIGN(32) double w00_arr[4];
      SCIPY_ALIGN(32) double w01_arr[4];
      SCIPY_ALIGN(32) double w10_arr[4];
      SCIPY_ALIGN(32) double w11_arr[4];

      _mm_storeu_si128((__m128i *)xi_arr, xi);
      _mm_storeu_si128((__m128i *)yi_arr, yi);
      _mm256_store_pd(w00_arr, w00);
      _mm256_store_pd(w01_arr, w01);
      _mm256_store_pd(w10_arr, w10);
      _mm256_store_pd(w11_arr, w11);

      SCIPY_ALIGN(32) double result[4];

      /* Process each pixel - with bounds checking */
      for (int j = 0; j < 4; j++) {
        int x0 = xi_arr[j];
        int y0 = yi_arr[j];

        /* Bilinear interpolation with correct axis convention */
        /* x0 is column index, y0 is row index */

        /* Check bounds for each of the 4 corners and get values */
        double v00, v01, v10, v11;

        /* Corner (y0, x0) - row y0, col x0 */
        if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
          v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
        } else {
          v00 = cval;
        }

        /* Corner (y0, x0+1) - row y0, col x0+1 */
        if (y0 >= 0 && y0 < H && (x0 + 1) >= 0 && (x0 + 1) < W) {
          v01 = in_data[y0 * in_stride_y + (x0 + 1) * in_stride_x];
        } else {
          v01 = cval;
        }

        /* Corner (y0+1, x0) - row y0+1, col x0 */
        if ((y0 + 1) >= 0 && (y0 + 1) < H && x0 >= 0 && x0 < W) {
          v10 = in_data[(y0 + 1) * in_stride_y + x0 * in_stride_x];
        } else {
          v10 = cval;
        }

        /* Corner (y0+1, x0+1) - row y0+1, col x0+1 */
        if ((y0 + 1) >= 0 && (y0 + 1) < H && (x0 + 1) >= 0 && (x0 + 1) < W) {
          v11 = in_data[(y0 + 1) * in_stride_y + (x0 + 1) * in_stride_x];
        } else {
          v11 = cval;
        }

        /* Bilinear interpolation using precomputed weights */
        result[j] = v00 * w00_arr[j] + v01 * w01_arr[j] + v10 * w10_arr[j] +
                    v11 * w11_arr[j];
      }

      /* Store results */
      __m256d vresult = _mm256_load_pd(result);
      _mm256_storeu_pd(out_row + ox * out_stride_x, vresult);
    }

    /* Scalar cleanup for remaining pixels */
    for (; ox < OW; ox++) {
      /* Scalar cleanup with correct SciPy axis convention */
      double y_src = m01 * (double)ox + base_y; /* rows */
      double x_src = m11 * (double)ox + base_x; /* cols */

      int x0 = (int)floor(x_src); // column index
      int y0 = (int)floor(y_src); // row index

      double fx = x_src - x0;
      double fy = y_src - y0;

      /* Check bounds for each of the 4 corners and get values */
      double v00, v01, v10, v11;

      /* Corner (y0, x0) - row y0, col x0 */
      if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
        v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
      } else {
        v00 = cval;
      }

      /* Corner (y0, x0+1) - row y0, col x0+1 */
      if (y0 >= 0 && y0 < H && (x0 + 1) >= 0 && (x0 + 1) < W) {
        v01 = in_data[y0 * in_stride_y + (x0 + 1) * in_stride_x];
      } else {
        v01 = cval;
      }

      /* Corner (y0+1, x0) - row y0+1, col x0 */
      if ((y0 + 1) >= 0 && (y0 + 1) < H && x0 >= 0 && x0 < W) {
        v10 = in_data[(y0 + 1) * in_stride_y + x0 * in_stride_x];
      } else {
        v10 = cval;
      }

      /* Corner (y0+1, x0+1) - row y0+1, col x0+1 */
      if ((y0 + 1) >= 0 && (y0 + 1) < H && (x0 + 1) >= 0 && (x0 + 1) < W) {
        v11 = in_data[(y0 + 1) * in_stride_y + (x0 + 1) * in_stride_x];
      } else {
        v11 = cval;
      }

      out_row[ox * out_stride_x] = v00 * (1 - fx) * (1 - fy) +
                                   v01 * fx * (1 - fy) + v10 * (1 - fx) * fy +
                                   v11 * fx * fy;
    }
  }

  return 1;
}

/* AVX2 optimized 3D trilinear interpolation for float64 */
int NI_GeometricTransform_3D_trilinear_f64_avx2(
    PyArrayObject *input, PyArrayObject *output, const double *M,
    const double *shift, int nprepad, int mode, double cval) {
  const npy_intp D = PyArray_DIM(input, 0); /* Depth */
  const npy_intp H = PyArray_DIM(input, 1); /* Height */
  const npy_intp W = PyArray_DIM(input, 2); /* Width */
  const npy_intp OD = PyArray_DIM(output, 0);
  const npy_intp OH = PyArray_DIM(output, 1);
  const npy_intp OW = PyArray_DIM(output, 2);
  const double *in_data = (const double *)PyArray_DATA(input);
  double *out_data = (double *)PyArray_DATA(output);
  const npy_intp in_stride_z = PyArray_STRIDE(input, 0) / sizeof(double);
  const npy_intp in_stride_y = PyArray_STRIDE(input, 1) / sizeof(double);
  const npy_intp in_stride_x = PyArray_STRIDE(input, 2) / sizeof(double);
  const npy_intp out_stride_z = PyArray_STRIDE(output, 0) / sizeof(double);
  const npy_intp out_stride_y = PyArray_STRIDE(output, 1) / sizeof(double);
  const npy_intp out_stride_x = PyArray_STRIDE(output, 2) / sizeof(double);

  /* Reject tiny volumes and check for potential overflow */
  if (D < 2 || H < 2 || W < 2)
    return 0;
  if (D > INT_MAX - 1 || H > INT_MAX - 1 || W > INT_MAX - 1)
    return 0; /* Prevent int overflow */

  /* 3x3 matrix and shift */
  const double m00 = M[0], m01 = M[1], m02 = M[2];
  const double m10 = M[3], m11 = M[4], m12 = M[5];
  const double m20 = M[6], m21 = M[7], m22 = M[8];
  const double shift_z = shift[0] + nprepad;
  const double shift_y = shift[1] + nprepad;
  const double shift_x = shift[2] + nprepad;

  const __m256d one = _mm256_set1_pd(1.0);

/* Process output voxels */
#pragma omp parallel for collapse(2) schedule(static)
  for (npy_intp oz = 0; oz < OD; ++oz) {
    for (npy_intp oy = 0; oy < OH; ++oy) {
      double *out_row = out_data + oz * out_stride_z + oy * out_stride_y;

      /* Base source coordinates for this row (SciPy: matrix * [oz, oy, ox]) */
      double base_z = m00 * (double)oz + m01 * (double)oy + shift_z;
      double base_y = m10 * (double)oz + m11 * (double)oy + shift_y;
      double base_x = m20 * (double)oz + m21 * (double)oy + shift_x;

      npy_intp ox = 0;

      /* AVX2 loop - process 4 voxels at a time */
      for (; ox + 3 < OW; ox += 4) {
        /* Generate x indices */
        __m256d vx = _mm256_setr_pd((double)ox, (double)(ox + 1),
                                    (double)(ox + 2), (double)(ox + 3));

        /* Compute source coordinates (matrix * [oz, oy, ox]) */
        __m256d zs =
            _mm256_fmadd_pd(_mm256_set1_pd(m02), vx, _mm256_set1_pd(base_z));
        __m256d ys =
            _mm256_fmadd_pd(_mm256_set1_pd(m12), vx, _mm256_set1_pd(base_y));
        __m256d xs =
            _mm256_fmadd_pd(_mm256_set1_pd(m22), vx, _mm256_set1_pd(base_x));

        /* Floor to get integer coordinates */
        __m256d z_floor = _mm256_floor_pd(zs);
        __m256d y_floor = _mm256_floor_pd(ys);
        __m256d x_floor = _mm256_floor_pd(xs);

        /* Get fractional parts */
        __m256d fz = _mm256_sub_pd(zs, z_floor);
        __m256d fy = _mm256_sub_pd(ys, y_floor);
        __m256d fx = _mm256_sub_pd(xs, x_floor);

        /* Compute trilinear weights (8 corners) */
        __m256d one_minus_fx = _mm256_sub_pd(one, fx);
        __m256d one_minus_fy = _mm256_sub_pd(one, fy);
        __m256d one_minus_fz = _mm256_sub_pd(one, fz);

        __m256d w000 = _mm256_mul_pd(_mm256_mul_pd(one_minus_fx, one_minus_fy),
                                     one_minus_fz);
        __m256d w001 =
            _mm256_mul_pd(_mm256_mul_pd(fx, one_minus_fy), one_minus_fz);
        __m256d w010 =
            _mm256_mul_pd(_mm256_mul_pd(one_minus_fx, fy), one_minus_fz);
        __m256d w011 = _mm256_mul_pd(_mm256_mul_pd(fx, fy), one_minus_fz);
        __m256d w100 =
            _mm256_mul_pd(_mm256_mul_pd(one_minus_fx, one_minus_fy), fz);
        __m256d w101 = _mm256_mul_pd(_mm256_mul_pd(fx, one_minus_fy), fz);
        __m256d w110 = _mm256_mul_pd(_mm256_mul_pd(one_minus_fx, fy), fz);
        __m256d w111 = _mm256_mul_pd(_mm256_mul_pd(fx, fy), fz);

        /* Convert to integers and extract */
        __m128i zi = _mm256_cvttpd_epi32(z_floor);
        __m128i yi = _mm256_cvttpd_epi32(y_floor);
        __m128i xi = _mm256_cvttpd_epi32(x_floor);

        SCIPY_ALIGN(32) int xi_arr[4];
        SCIPY_ALIGN(32) int yi_arr[4];
        SCIPY_ALIGN(32) int zi_arr[4];
        SCIPY_ALIGN(32) double weights[8][4];

        _mm_storeu_si128((__m128i *)zi_arr, zi);
        _mm_storeu_si128((__m128i *)yi_arr, yi);
        _mm_storeu_si128((__m128i *)xi_arr, xi);

        _mm256_store_pd(weights[0], w000);
        _mm256_store_pd(weights[1], w001);
        _mm256_store_pd(weights[2], w010);
        _mm256_store_pd(weights[3], w011);
        _mm256_store_pd(weights[4], w100);
        _mm256_store_pd(weights[5], w101);
        _mm256_store_pd(weights[6], w110);
        _mm256_store_pd(weights[7], w111);

        SCIPY_ALIGN(32) double result[4];

        /* Process each voxel */
        for (int j = 0; j < 4; j++) {
          int z0 = zi_arr[j];
          int y0 = yi_arr[j];
          int x0 = xi_arr[j];

          /* Check bounds */
          if (x0 >= 0 && x0 < W - 1 && y0 >= 0 && y0 < H - 1 && z0 >= 0 &&
              z0 < D - 1) {

            /* Compute indices for 8 corners */
            npy_intp idx000 =
                z0 * in_stride_z + y0 * in_stride_y + x0 * in_stride_x;
            npy_intp idx001 = idx000 + in_stride_x;
            npy_intp idx010 = idx000 + in_stride_y;
            npy_intp idx011 = idx010 + in_stride_x;
            npy_intp idx100 = idx000 + in_stride_z;
            npy_intp idx101 = idx100 + in_stride_x;
            npy_intp idx110 = idx100 + in_stride_y;
            npy_intp idx111 = idx110 + in_stride_x;

/* Prefetch further ahead along x for both z-planes */
#if USE_PREFETCH
            const int PX = 16; /* 16 doubles = 128 bytes ahead */
            if (x0 + PX < W - 1) {
              _mm_prefetch((const char *)(in_data + idx000 + PX * in_stride_x),
                           _MM_HINT_T0);
              _mm_prefetch((const char *)(in_data + idx100 + PX * in_stride_x),
                           _MM_HINT_T0);
            }
#endif

            /* Load 8 corner values */
            double v000 = in_data[idx000];
            double v001 = in_data[idx001];
            double v010 = in_data[idx010];
            double v011 = in_data[idx011];
            double v100 = in_data[idx100];
            double v101 = in_data[idx101];
            double v110 = in_data[idx110];
            double v111 = in_data[idx111];

            /* Trilinear interpolation */
            result[j] = v000 * weights[0][j] + v001 * weights[1][j] +
                        v010 * weights[2][j] + v011 * weights[3][j] +
                        v100 * weights[4][j] + v101 * weights[5][j] +
                        v110 * weights[6][j] + v111 * weights[7][j];
          } else {
            result[j] = cval;
          }
        }

        /* Store results */
        __m256d vresult = _mm256_load_pd(result);
        _mm256_storeu_pd(out_row + ox * out_stride_x, vresult);
      }

      /* Scalar cleanup */
      for (; ox < OW; ox++) {
        double z_src = m02 * (double)ox + base_z;
        double y_src = m12 * (double)ox + base_y;
        double x_src = m22 * (double)ox + base_x;

        int z0 = (int)floor(z_src);
        int y0 = (int)floor(y_src);
        int x0 = (int)floor(x_src);

        if (x0 >= 0 && x0 < W - 1 && y0 >= 0 && y0 < H - 1 && z0 >= 0 &&
            z0 < D - 1) {

          double fx = x_src - x0;
          double fy = y_src - y0;
          double fz = z_src - z0;

          /* 8 corners for trilinear */
          npy_intp idx000 =
              z0 * in_stride_z + y0 * in_stride_y + x0 * in_stride_x;
          npy_intp idx001 = idx000 + in_stride_x;
          npy_intp idx010 = idx000 + in_stride_y;
          npy_intp idx011 = idx010 + in_stride_x;
          npy_intp idx100 = idx000 + in_stride_z;
          npy_intp idx101 = idx100 + in_stride_x;
          npy_intp idx110 = idx100 + in_stride_y;
          npy_intp idx111 = idx110 + in_stride_x;

          double v000 = in_data[idx000];
          double v001 = in_data[idx001];
          double v010 = in_data[idx010];
          double v011 = in_data[idx011];
          double v100 = in_data[idx100];
          double v101 = in_data[idx101];
          double v110 = in_data[idx110];
          double v111 = in_data[idx111];

          out_row[ox * out_stride_x] =
              v000 * (1 - fx) * (1 - fy) * (1 - fz) +
              v001 * fx * (1 - fy) * (1 - fz) +
              v010 * (1 - fx) * fy * (1 - fz) + v011 * fx * fy * (1 - fz) +
              v100 * (1 - fx) * (1 - fy) * fz + v101 * fx * (1 - fy) * fz +
              v110 * (1 - fx) * fy * fz + v111 * fx * fy * fz;
        } else {
          out_row[ox * out_stride_x] = cval;
        }
      }
    }
  }

  return 1;
}

/* AVX2 optimized 2D bilinear for float32 - processes 8 floats at a time */
int NI_GeometricTransform_2D_bilinear_f32_avx2(PyArrayObject *input,
                                               PyArrayObject *output,
                                               const double *M,
                                               const double *shift, int nprepad,
                                               int mode, double cval) {
  const npy_intp H = PyArray_DIM(input, 0);
  const npy_intp W = PyArray_DIM(input, 1);
  const npy_intp OH = PyArray_DIM(output, 0);
  const npy_intp OW = PyArray_DIM(output, 1);
  const float *in_data = (const float *)PyArray_DATA(input);
  float *out_data = (float *)PyArray_DATA(output);
  const npy_intp in_stride_y = PyArray_STRIDE(input, 0) / sizeof(float);
  const npy_intp in_stride_x = PyArray_STRIDE(input, 1) / sizeof(float);
  const npy_intp out_stride_y = PyArray_STRIDE(output, 0) / sizeof(float);
  const npy_intp out_stride_x = PyArray_STRIDE(output, 1) / sizeof(float);

  /* Reject tiny images and check for potential overflow */
  if (H < 2 || W < 2)
    return 0;
  if (H > INT_MAX - 1 || W > INT_MAX - 1)
    return 0; /* Prevent int overflow */

  /* Matrix and shift as floats - SciPy axis convention */
  const float m00 = (float)M[0],
              m01 = (float)M[1]; /* contributes to y_src (axis 0) */
  const float m10 = (float)M[2],
              m11 = (float)M[3]; /* contributes to x_src (axis 1) */
  const float shy = (float)(shift[0] + nprepad); /* shift for rows (y) */
  const float shx = (float)(shift[1] + nprepad); /* shift for cols (x) */

  const __m256 one = _mm256_set1_ps(1.0f);

/* Process output pixels */
#pragma omp parallel for schedule(static)
  for (npy_intp oy = 0; oy < OH; ++oy) {
    float *out_row = out_data + oy * out_stride_y;

    /* SciPy: y_src uses row 0 of M + shift[0]; x_src uses row 1 + shift[1] */
    const float base_y = m00 * (float)oy + shy; /* y_src base */
    const float base_x = m10 * (float)oy + shx; /* x_src base */

    npy_intp ox = 0;

    /* AVX2 loop - process 8 pixels at a time */
    for (; ox + 7 < OW; ox += 8) {
      /* Generate x indices [ox, ox+1, ..., ox+7] */
      __m256 vx = _mm256_setr_ps(
          (float)ox, (float)(ox + 1), (float)(ox + 2), (float)(ox + 3),
          (float)(ox + 4), (float)(ox + 5), (float)(ox + 6), (float)(ox + 7));

      /* Compute source coordinates using SciPy axis convention */
      __m256 ys = _mm256_fmadd_ps(_mm256_set1_ps(m01), vx,
                                  _mm256_set1_ps(base_y)); /* y_src */
      __m256 xs = _mm256_fmadd_ps(_mm256_set1_ps(m11), vx,
                                  _mm256_set1_ps(base_x)); /* x_src */

      /* Floor to get integer coordinates */
      __m256 x_floor = _mm256_floor_ps(xs);
      __m256 y_floor = _mm256_floor_ps(ys);

      /* Get fractional parts */
      __m256 fx = _mm256_sub_ps(xs, x_floor);
      __m256 fy = _mm256_sub_ps(ys, y_floor);

      /* Compute bilinear weights */
      __m256 one_minus_fx = _mm256_sub_ps(one, fx);
      __m256 one_minus_fy = _mm256_sub_ps(one, fy);

      __m256 w00 = _mm256_mul_ps(one_minus_fx, one_minus_fy);
      __m256 w01 = _mm256_mul_ps(fx, one_minus_fy);
      __m256 w10 = _mm256_mul_ps(one_minus_fx, fy);
      __m256 w11 = _mm256_mul_ps(fx, fy);

      /* Convert to integers */
      __m256i xi = _mm256_cvttps_epi32(x_floor);
      __m256i yi = _mm256_cvttps_epi32(y_floor);

      /* Extract and process */
      SCIPY_ALIGN(32) int xi_arr[8];
      SCIPY_ALIGN(32) int yi_arr[8];
      SCIPY_ALIGN(32) float w00_arr[8];
      SCIPY_ALIGN(32) float w01_arr[8];
      SCIPY_ALIGN(32) float w10_arr[8];
      SCIPY_ALIGN(32) float w11_arr[8];

      _mm256_storeu_si256((__m256i *)xi_arr, xi);
      _mm256_storeu_si256((__m256i *)yi_arr, yi);
      _mm256_store_ps(w00_arr, w00);
      _mm256_store_ps(w01_arr, w01);
      _mm256_store_ps(w10_arr, w10);
      _mm256_store_ps(w11_arr, w11);

      SCIPY_ALIGN(32) float result[8];

      /* Process each pixel - with bounds checking */
      for (int j = 0; j < 8; j++) {
        int x0 = xi_arr[j];
        int y0 = yi_arr[j];

        /* Check bounds for each of the 4 corners individually */
        float v00, v01, v10, v11;

        /* Corner (x0, y0) - row x0, col y0 */
        if (x0 >= 0 && x0 < H && y0 >= 0 && y0 < W) {
          v00 = in_data[x0 * in_stride_y + y0 * in_stride_x];
        } else {
          v00 = (float)cval;
        }

        /* Corner (x0, y0+1) - row x0, col y0+1 */
        if (x0 >= 0 && x0 < H && (y0 + 1) >= 0 && (y0 + 1) < W) {
          v01 = in_data[x0 * in_stride_y + (y0 + 1) * in_stride_x];
        } else {
          v01 = (float)cval;
        }

        /* Corner (x0+1, y0) - row x0+1, col y0 */
        if ((x0 + 1) >= 0 && (x0 + 1) < H && y0 >= 0 && y0 < W) {
          v10 = in_data[(x0 + 1) * in_stride_y + y0 * in_stride_x];
        } else {
          v10 = (float)cval;
        }

        /* Corner (x0+1, y0+1) - row x0+1, col y0+1 */
        if ((x0 + 1) >= 0 && (x0 + 1) < H && (y0 + 1) >= 0 && (y0 + 1) < W) {
          v11 = in_data[(x0 + 1) * in_stride_y + (y0 + 1) * in_stride_x];
        } else {
          v11 = (float)cval;
        }

/* Prefetch further ahead along x (32 floats = 128 bytes ahead) */
#if USE_PREFETCH
        const int PX = 32; /* Tunable: try 16-64 */
        if (x0 >= 0 && x0 + PX < W && y0 >= 0 && y0 < H) {
          _mm_prefetch((const char *)(in_data + y0 * in_stride_y +
                                      (x0 + PX) * in_stride_x),
                       _MM_HINT_T0);
          if (y0 + 1 < H) {
            _mm_prefetch((const char *)(in_data + (y0 + 1) * in_stride_y +
                                        (x0 + PX) * in_stride_x),
                         _MM_HINT_T0);
          }
        }
#endif

        /* Bilinear interpolation using precomputed weights */
        result[j] = v00 * w00_arr[j] + v01 * w01_arr[j] + v10 * w10_arr[j] +
                    v11 * w11_arr[j];
      }

      /* Store results */
      __m256 vresult = _mm256_load_ps(result);
      _mm256_storeu_ps(out_row + ox * out_stride_x, vresult);
    }

    /* Scalar cleanup */
    for (; ox < OW; ox++) {
      /* Scalar cleanup with correct SciPy axis convention */
      float y_src = m01 * (float)ox + base_y; /* rows */
      float x_src = m11 * (float)ox + base_x; /* cols */

      int x0 = (int)floorf(x_src); // column index
      int y0 = (int)floorf(y_src); // row index

      float fx = x_src - x0;
      float fy = y_src - y0;

      /* Check bounds for each of the 4 corners individually */
      float v00, v01, v10, v11;

      /* Corner (y0, x0) */
      if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
        v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
      } else {
        v00 = (float)cval;
      }

      /* Corner (y0, x0+1) */
      if (y0 >= 0 && y0 < H && (x0 + 1) >= 0 && (x0 + 1) < W) {
        v01 = in_data[y0 * in_stride_y + (x0 + 1) * in_stride_x];
      } else {
        v01 = (float)cval;
      }

      /* Corner (y0+1, x0) */
      if ((y0 + 1) >= 0 && (y0 + 1) < H && x0 >= 0 && x0 < W) {
        v10 = in_data[(y0 + 1) * in_stride_y + x0 * in_stride_x];
      } else {
        v10 = (float)cval;
      }

      /* Corner (y0+1, x0+1) */
      if ((y0 + 1) >= 0 && (y0 + 1) < H && (x0 + 1) >= 0 && (x0 + 1) < W) {
        v11 = in_data[(y0 + 1) * in_stride_y + (x0 + 1) * in_stride_x];
      } else {
        v11 = (float)cval;
      }

      out_row[ox * out_stride_x] = v00 * (1 - fx) * (1 - fy) +
                                   v01 * fx * (1 - fy) + v10 * (1 - fx) * fy +
                                   v11 * fx * fy;
    }
  }

  return 1;
}

/* AVX2 optimized 3D trilinear interpolation for float32 */
int NI_GeometricTransform_3D_trilinear_f32_avx2(
    PyArrayObject *input, PyArrayObject *output, const double *M,
    const double *shift, int nprepad, int mode, double cval) {
  const npy_intp D = PyArray_DIM(input, 0); /* Depth */
  const npy_intp H = PyArray_DIM(input, 1); /* Height */
  const npy_intp W = PyArray_DIM(input, 2); /* Width */
  const npy_intp OD = PyArray_DIM(output, 0);
  const npy_intp OH = PyArray_DIM(output, 1);
  const npy_intp OW = PyArray_DIM(output, 2);
  const float *in_data = (const float *)PyArray_DATA(input);
  float *out_data = (float *)PyArray_DATA(output);
  const npy_intp in_stride_z = PyArray_STRIDE(input, 0) / sizeof(float);
  const npy_intp in_stride_y = PyArray_STRIDE(input, 1) / sizeof(float);
  const npy_intp in_stride_x = PyArray_STRIDE(input, 2) / sizeof(float);
  const npy_intp out_stride_z = PyArray_STRIDE(output, 0) / sizeof(float);
  const npy_intp out_stride_y = PyArray_STRIDE(output, 1) / sizeof(float);
  const npy_intp out_stride_x = PyArray_STRIDE(output, 2) / sizeof(float);

  /* Reject tiny volumes and check for potential overflow */
  if (D < 2 || H < 2 || W < 2)
    return 0;
  if (D > INT_MAX - 1 || H > INT_MAX - 1 || W > INT_MAX - 1)
    return 0; /* Prevent int overflow */

  /* 3x3 matrix and shift */
  const float m00 = (float)M[0], m01 = (float)M[1], m02 = (float)M[2];
  const float m10 = (float)M[3], m11 = (float)M[4], m12 = (float)M[5];
  const float m20 = (float)M[6], m21 = (float)M[7], m22 = (float)M[8];
  const float shift_z = (float)(shift[0] + nprepad);
  const float shift_y = (float)(shift[1] + nprepad);
  const float shift_x = (float)(shift[2] + nprepad);

  const __m256 one = _mm256_set1_ps(1.0f);

/* Process output voxels */
#pragma omp parallel for collapse(2) schedule(static)
  for (npy_intp oz = 0; oz < OD; ++oz) {
    for (npy_intp oy = 0; oy < OH; ++oy) {
      float *out_row = out_data + oz * out_stride_z + oy * out_stride_y;

      /* Base source coordinates for this row (SciPy: matrix * [oz, oy, ox]) */
      float base_z = m00 * (float)oz + m01 * (float)oy + shift_z;
      float base_y = m10 * (float)oz + m11 * (float)oy + shift_y;
      float base_x = m20 * (float)oz + m21 * (float)oy + shift_x;

      npy_intp ox = 0;

      /* AVX2 loop - process 8 voxels at a time */
      for (; ox + 7 < OW; ox += 8) {
        /* Generate x indices */
        __m256 vx = _mm256_setr_ps(
            (float)ox, (float)(ox + 1), (float)(ox + 2), (float)(ox + 3),
            (float)(ox + 4), (float)(ox + 5), (float)(ox + 6), (float)(ox + 7));

        /* Compute source coordinates (matrix * [oz, oy, ox]) */
        __m256 zs =
            _mm256_fmadd_ps(_mm256_set1_ps(m02), vx, _mm256_set1_ps(base_z));
        __m256 ys =
            _mm256_fmadd_ps(_mm256_set1_ps(m12), vx, _mm256_set1_ps(base_y));
        __m256 xs =
            _mm256_fmadd_ps(_mm256_set1_ps(m22), vx, _mm256_set1_ps(base_x));

        /* Floor to get integer coordinates */
        __m256 z_floor = _mm256_floor_ps(zs);
        __m256 y_floor = _mm256_floor_ps(ys);
        __m256 x_floor = _mm256_floor_ps(xs);

        /* Get fractional parts */
        __m256 fz = _mm256_sub_ps(zs, z_floor);
        __m256 fy = _mm256_sub_ps(ys, y_floor);
        __m256 fx = _mm256_sub_ps(xs, x_floor);

        /* Compute trilinear weights (8 corners) */
        __m256 one_minus_fx = _mm256_sub_ps(one, fx);
        __m256 one_minus_fy = _mm256_sub_ps(one, fy);
        __m256 one_minus_fz = _mm256_sub_ps(one, fz);

        __m256 w000 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, one_minus_fy),
                                    one_minus_fz);
        __m256 w001 =
            _mm256_mul_ps(_mm256_mul_ps(fx, one_minus_fy), one_minus_fz);
        __m256 w010 =
            _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, fy), one_minus_fz);
        __m256 w011 = _mm256_mul_ps(_mm256_mul_ps(fx, fy), one_minus_fz);
        __m256 w100 =
            _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, one_minus_fy), fz);
        __m256 w101 = _mm256_mul_ps(_mm256_mul_ps(fx, one_minus_fy), fz);
        __m256 w110 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, fy), fz);
        __m256 w111 = _mm256_mul_ps(_mm256_mul_ps(fx, fy), fz);

        /* Convert to integers */
        __m256i zi = _mm256_cvttps_epi32(z_floor);
        __m256i yi = _mm256_cvttps_epi32(y_floor);
        __m256i xi = _mm256_cvttps_epi32(x_floor);

        /* Extract values */
        SCIPY_ALIGN(32) int xi_arr[8];
        SCIPY_ALIGN(32) int yi_arr[8];
        SCIPY_ALIGN(32) int zi_arr[8];
        SCIPY_ALIGN(32) float weights[8][8]; /* 8 corners x 8 voxels */

        _mm256_storeu_si256((__m256i *)zi_arr, zi);
        _mm256_storeu_si256((__m256i *)yi_arr, yi);
        _mm256_storeu_si256((__m256i *)xi_arr, xi);

        _mm256_store_ps(weights[0], w000);
        _mm256_store_ps(weights[1], w001);
        _mm256_store_ps(weights[2], w010);
        _mm256_store_ps(weights[3], w011);
        _mm256_store_ps(weights[4], w100);
        _mm256_store_ps(weights[5], w101);
        _mm256_store_ps(weights[6], w110);
        _mm256_store_ps(weights[7], w111);

        SCIPY_ALIGN(32) float result[8];

        /* Process each voxel */
        for (int j = 0; j < 8; j++) {
          int z0 = zi_arr[j];
          int y0 = yi_arr[j];
          int x0 = xi_arr[j];

          /* Check bounds */
          if (x0 >= 0 && x0 < W - 1 && y0 >= 0 && y0 < H - 1 && z0 >= 0 &&
              z0 < D - 1) {

            /* Compute indices for 8 corners */
            npy_intp idx000 =
                z0 * in_stride_z + y0 * in_stride_y + x0 * in_stride_x;
            npy_intp idx001 = idx000 + in_stride_x;
            npy_intp idx010 = idx000 + in_stride_y;
            npy_intp idx011 = idx010 + in_stride_x;
            npy_intp idx100 = idx000 + in_stride_z;

/* Prefetch further ahead along x for both z-planes */
#if USE_PREFETCH
            const int PX = 32; /* 32 floats = 128 bytes ahead */
            if (x0 + PX < W - 1) {
              _mm_prefetch((const char *)(in_data + idx000 + PX * in_stride_x),
                           _MM_HINT_T0);
              _mm_prefetch((const char *)(in_data + idx100 + PX * in_stride_x),
                           _MM_HINT_T0);
            }
#endif
            npy_intp idx101 = idx100 + in_stride_x;
            npy_intp idx110 = idx100 + in_stride_y;
            npy_intp idx111 = idx110 + in_stride_x;

            /* Load 8 corner values */
            float v000 = in_data[idx000];
            float v001 = in_data[idx001];
            float v010 = in_data[idx010];
            float v011 = in_data[idx011];
            float v100 = in_data[idx100];
            float v101 = in_data[idx101];
            float v110 = in_data[idx110];
            float v111 = in_data[idx111];

            /* Trilinear interpolation */
            result[j] = v000 * weights[0][j] + v001 * weights[1][j] +
                        v010 * weights[2][j] + v011 * weights[3][j] +
                        v100 * weights[4][j] + v101 * weights[5][j] +
                        v110 * weights[6][j] + v111 * weights[7][j];
          } else {
            result[j] = (float)cval;
          }
        }

        /* Store results */
        __m256 vresult = _mm256_load_ps(result);
        _mm256_storeu_ps(out_row + ox * out_stride_x, vresult);
      }

      /* Scalar cleanup */
      for (; ox < OW; ox++) {
        float z_src = m02 * (float)ox + base_z;
        float y_src = m12 * (float)ox + base_y;
        float x_src = m22 * (float)ox + base_x;

        int z0 = (int)floorf(z_src);
        int y0 = (int)floorf(y_src);
        int x0 = (int)floorf(x_src);

        if (x0 >= 0 && x0 < W - 1 && y0 >= 0 && y0 < H - 1 && z0 >= 0 &&
            z0 < D - 1) {

          float fx = x_src - x0;
          float fy = y_src - y0;
          float fz = z_src - z0;

          /* 8 corners for trilinear */
          npy_intp idx000 =
              z0 * in_stride_z + y0 * in_stride_y + x0 * in_stride_x;
          npy_intp idx001 = idx000 + in_stride_x;
          npy_intp idx010 = idx000 + in_stride_y;
          npy_intp idx011 = idx010 + in_stride_x;
          npy_intp idx100 = idx000 + in_stride_z;
          npy_intp idx101 = idx100 + in_stride_x;
          npy_intp idx110 = idx100 + in_stride_y;
          npy_intp idx111 = idx110 + in_stride_x;

          float v000 = in_data[idx000];
          float v001 = in_data[idx001];
          float v010 = in_data[idx010];
          float v011 = in_data[idx011];
          float v100 = in_data[idx100];
          float v101 = in_data[idx101];
          float v110 = in_data[idx110];
          float v111 = in_data[idx111];

          out_row[ox * out_stride_x] =
              v000 * (1 - fx) * (1 - fy) * (1 - fz) +
              v001 * fx * (1 - fy) * (1 - fz) +
              v010 * (1 - fx) * fy * (1 - fz) + v011 * fx * fy * (1 - fz) +
              v100 * (1 - fx) * (1 - fy) * fz + v101 * fx * (1 - fy) * fz +
              v110 * (1 - fx) * fy * fz + v111 * fx * fy * fz;
        } else {
          out_row[ox * out_stride_x] = (float)cval;
        }
      }
    }
  }

  return 1;
}

/* AVX2 optimized 2D bilinear interpolation for float16 (FP16) */
int NI_GeometricTransform_2D_bilinear_f16_avx2(PyArrayObject *input,
                                               PyArrayObject *output,
                                               const double *M,
                                               const double *shift, int nprepad,
                                               int mode, double cval) {
  const npy_intp H = PyArray_DIM(input, 0);
  const npy_intp W = PyArray_DIM(input, 1);
  const npy_intp OH = PyArray_DIM(output, 0);
  const npy_intp OW = PyArray_DIM(output, 1);
  const uint16_t *in_data =
      (const uint16_t *)PyArray_DATA(input); /* FP16 as uint16_t */
  uint16_t *out_data = (uint16_t *)PyArray_DATA(output);
  const npy_intp in_stride_y = PyArray_STRIDE(input, 0) / sizeof(uint16_t);
  const npy_intp in_stride_x = PyArray_STRIDE(input, 1) / sizeof(uint16_t);
  const npy_intp out_stride_y = PyArray_STRIDE(output, 0) / sizeof(uint16_t);
  const npy_intp out_stride_x = PyArray_STRIDE(output, 1) / sizeof(uint16_t);

  /* Reject tiny images and check for potential overflow */
  if (H < 2 || W < 2)
    return 0;
  if (H > INT_MAX - 1 || W > INT_MAX - 1)
    return 0; /* Prevent int overflow */

  /* Matrix and shift as floats - SciPy axis convention */
  const float m00 = (float)M[0],
              m01 = (float)M[1]; /* contributes to y_src (axis 0) */
  const float m10 = (float)M[2],
              m11 = (float)M[3]; /* contributes to x_src (axis 1) */
  const float shy = (float)(shift[0] + nprepad); /* shift for rows (y) */
  const float shx = (float)(shift[1] + nprepad); /* shift for cols (x) */

  const __m256 one = _mm256_set1_ps(1.0f);
  /* Convert cval to FP16 using F16C intrinsics */
  __m256 cval_f32_vec = _mm256_set1_ps((float)cval);
  __m128i cval_f16_vec = _mm256_cvtps_ph(cval_f32_vec, 0);
  const uint16_t cval_fp16 = _mm_extract_epi16(cval_f16_vec, 0);

  /* Process output pixels */
  #pragma omp parallel for schedule(static)
  for (npy_intp oy = 0; oy < OH; ++oy) {
    uint16_t *out_row = out_data + oy * out_stride_y;

    /* SciPy: y_src uses row 0 of M + shift[0]; x_src uses row 1 + shift[1] */
    const float base_y = m00 * (float)oy + shy; /* y_src base */
    const float base_x = m10 * (float)oy + shx; /* x_src base */

    npy_intp ox = 0;

    /* AVX2 loop - process 8 pixels at a time with optimized FP16 handling */
    for (; ox + 7 < OW; ox += 8) {
      /* Generate x indices [ox, ox+1, ..., ox+7] */
      __m256 vx = _mm256_setr_ps(
          (float)ox, (float)(ox + 1), (float)(ox + 2), (float)(ox + 3),
          (float)(ox + 4), (float)(ox + 5), (float)(ox + 6), (float)(ox + 7));

      /* Compute source coordinates using SciPy axis convention */
      __m256 ys = _mm256_fmadd_ps(_mm256_set1_ps(m01), vx,
                                  _mm256_set1_ps(base_y)); /* y_src */
      __m256 xs = _mm256_fmadd_ps(_mm256_set1_ps(m11), vx,
                                  _mm256_set1_ps(base_x)); /* x_src */

      /* Floor to get integer coordinates */
      __m256 x_floor = _mm256_floor_ps(xs);
      __m256 y_floor = _mm256_floor_ps(ys);

      /* Get fractional parts */
      __m256 fx = _mm256_sub_ps(xs, x_floor);
      __m256 fy = _mm256_sub_ps(ys, y_floor);

      /* Compute bilinear weights */
      __m256 one_minus_fx = _mm256_sub_ps(one, fx);
      __m256 one_minus_fy = _mm256_sub_ps(one, fy);

      __m256 w00 = _mm256_mul_ps(one_minus_fx, one_minus_fy);
      __m256 w01 = _mm256_mul_ps(fx, one_minus_fy);
      __m256 w10 = _mm256_mul_ps(one_minus_fx, fy);
      __m256 w11 = _mm256_mul_ps(fx, fy);

      /* Convert to integers */
      __m256i xi = _mm256_cvttps_epi32(x_floor);
      __m256i yi = _mm256_cvttps_epi32(y_floor);

      /* Extract coordinate arrays */
      SCIPY_ALIGN(32) int xi_arr[8];
      SCIPY_ALIGN(32) int yi_arr[8];
      _mm256_storeu_si256((__m256i *)xi_arr, xi);
      _mm256_storeu_si256((__m256i *)yi_arr, yi);

      /* Separate corner arrays for optimal vectorization */
      SCIPY_ALIGN(16)
      uint16_t v00_fp16[8], v01_fp16[8], v10_fp16[8], v11_fp16[8];

      /* Gather corners with bounds checking - separated by corner type */
      for (int j = 0; j < 8; j++) {
        int x0 = xi_arr[j];
        int y0 = yi_arr[j];

        v00_fp16[j] = (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W)
                          ? in_data[y0 * in_stride_y + x0 * in_stride_x]
                          : cval_fp16;
        v01_fp16[j] = (y0 >= 0 && y0 < H && (x0 + 1) >= 0 && (x0 + 1) < W)
                          ? in_data[y0 * in_stride_y + (x0 + 1) * in_stride_x]
                          : cval_fp16;
        v10_fp16[j] = ((y0 + 1) >= 0 && (y0 + 1) < H && x0 >= 0 && x0 < W)
                          ? in_data[(y0 + 1) * in_stride_y + x0 * in_stride_x]
                          : cval_fp16;
        v11_fp16[j] =
            ((y0 + 1) >= 0 && (y0 + 1) < H && (x0 + 1) >= 0 && (x0 + 1) < W)
                ? in_data[(y0 + 1) * in_stride_y + (x0 + 1) * in_stride_x]
                : cval_fp16;
      }

      /* Convert each corner type from FP16 to FP32 */
      __m128i v00_i16 = _mm_loadu_si128((__m128i *)v00_fp16);
      __m128i v01_i16 = _mm_loadu_si128((__m128i *)v01_fp16);
      __m128i v10_i16 = _mm_loadu_si128((__m128i *)v10_fp16);
      __m128i v11_i16 = _mm_loadu_si128((__m128i *)v11_fp16);

      __m256 v00_f32 = _mm256_cvtph_ps(v00_i16);
      __m256 v01_f32 = _mm256_cvtph_ps(v01_i16);
      __m256 v10_f32 = _mm256_cvtph_ps(v10_i16);
      __m256 v11_f32 = _mm256_cvtph_ps(v11_i16);

      /* Fully vectorized bilinear interpolation using FMA */
      __m256 result_f32 = _mm256_fmadd_ps(
          v00_f32, w00,
          _mm256_fmadd_ps(
              v01_f32, w01,
              _mm256_fmadd_ps(v10_f32, w10, _mm256_mul_ps(v11_f32, w11))));

      /* Convert result back to FP16 and store directly */
      __m128i result_fp16 = _mm256_cvtps_ph(result_f32, 0);
      _mm_storeu_si128((__m128i *)(out_row + ox * out_stride_x), result_fp16);
    }

    /* Scalar cleanup */
    for (; ox < OW; ox++) {
      /* Scalar cleanup with correct SciPy axis convention */
      float y_src = m01 * (float)ox + base_y; /* rows */
      float x_src = m11 * (float)ox + base_x; /* cols */

      int x0 = (int)floorf(x_src); // column index
      int y0 = (int)floorf(y_src); // row index

      float fx = x_src - x0;
      float fy = y_src - y0;

      /* Check bounds for each of the 4 corners individually and get FP16 values
       */
      uint16_t fp16_v00, fp16_v01, fp16_v10, fp16_v11;

      /* Corner (y0, x0) */
      if (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W) {
        fp16_v00 = in_data[y0 * in_stride_y + x0 * in_stride_x];
      } else {
        fp16_v00 = cval_fp16;
      }

      /* Corner (y0, x0+1) */
      if (y0 >= 0 && y0 < H && (x0 + 1) >= 0 && (x0 + 1) < W) {
        fp16_v01 = in_data[y0 * in_stride_y + (x0 + 1) * in_stride_x];
      } else {
        fp16_v01 = cval_fp16;
      }

      /* Corner (y0+1, x0) */
      if ((y0 + 1) >= 0 && (y0 + 1) < H && x0 >= 0 && x0 < W) {
        fp16_v10 = in_data[(y0 + 1) * in_stride_y + x0 * in_stride_x];
      } else {
        fp16_v10 = cval_fp16;
      }

      /* Corner (y0+1, x0+1) */
      if ((y0 + 1) >= 0 && (y0 + 1) < H && (x0 + 1) >= 0 && (x0 + 1) < W) {
        fp16_v11 = in_data[(y0 + 1) * in_stride_y + (x0 + 1) * in_stride_x];
      } else {
        fp16_v11 = cval_fp16;
      }

      /* Optimized scalar path: convert 4 FP16 values at once */
      __m128i fp16_vec = _mm_setr_epi16(fp16_v00, fp16_v01, fp16_v10, fp16_v11,
                                        0, 0, 0, 0); /* Pad with zeros */
      __m256 fp32_vec = _mm256_cvtph_ps(fp16_vec);

      /* Extract corners directly using shuffle/blend instead of store/load */
      float v00 = _mm256_cvtss_f32(fp32_vec);
      float v01 = _mm256_cvtss_f32(_mm256_shuffle_ps(fp32_vec, fp32_vec, 1));
      float v10 = _mm256_cvtss_f32(_mm256_shuffle_ps(fp32_vec, fp32_vec, 2));
      float v11 = _mm256_cvtss_f32(_mm256_shuffle_ps(fp32_vec, fp32_vec, 3));

      /* Bilinear interpolation */
      float result_f32 = v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
                         v10 * (1 - fx) * fy + v11 * fx * fy;

      /* Convert single result back to FP16 efficiently */
      __m256 result_vec = _mm256_set1_ps(result_f32);
      __m128i result_fp16 = _mm256_cvtps_ph(result_vec, 0);
      out_row[ox * out_stride_x] = _mm_extract_epi16(result_fp16, 0);
    }
  }

  return 1;
}

/* AVX2 optimized 3D trilinear interpolation for float16 (FP16) */
int NI_GeometricTransform_3D_trilinear_f16_avx2(
    PyArrayObject *input, PyArrayObject *output, const double *M,
    const double *shift, int nprepad, int mode, double cval) {
  const npy_intp D = PyArray_DIM(input, 0); /* Depth */
  const npy_intp H = PyArray_DIM(input, 1); /* Height */
  const npy_intp W = PyArray_DIM(input, 2); /* Width */
  const npy_intp OD = PyArray_DIM(output, 0);
  const npy_intp OH = PyArray_DIM(output, 1);
  const npy_intp OW = PyArray_DIM(output, 2);
  const uint16_t *in_data =
      (const uint16_t *)PyArray_DATA(input); /* FP16 as uint16_t */
  uint16_t *out_data = (uint16_t *)PyArray_DATA(output);
  const npy_intp in_stride_z = PyArray_STRIDE(input, 0) / sizeof(uint16_t);
  const npy_intp in_stride_y = PyArray_STRIDE(input, 1) / sizeof(uint16_t);
  const npy_intp in_stride_x = PyArray_STRIDE(input, 2) / sizeof(uint16_t);
  const npy_intp out_stride_z = PyArray_STRIDE(output, 0) / sizeof(uint16_t);
  const npy_intp out_stride_y = PyArray_STRIDE(output, 1) / sizeof(uint16_t);
  const npy_intp out_stride_x = PyArray_STRIDE(output, 2) / sizeof(uint16_t);

  /* Reject tiny volumes and check for potential overflow */
  if (D < 2 || H < 2 || W < 2)
    return 0;
  if (D > INT_MAX - 1 || H > INT_MAX - 1 || W > INT_MAX - 1)
    return 0; /* Prevent int overflow */

  /* 3x3 matrix and shift */
  const float m00 = (float)M[0], m01 = (float)M[1], m02 = (float)M[2];
  const float m10 = (float)M[3], m11 = (float)M[4], m12 = (float)M[5];
  const float m20 = (float)M[6], m21 = (float)M[7], m22 = (float)M[8];
  const float shift_z = (float)(shift[0] + nprepad);
  const float shift_y = (float)(shift[1] + nprepad);
  const float shift_x = (float)(shift[2] + nprepad);

  const __m256 one = _mm256_set1_ps(1.0f);
  /* Convert cval to FP16 using F16C intrinsics */
  __m256 cval_f32_vec = _mm256_set1_ps((float)cval);
  __m128i cval_f16_vec = _mm256_cvtps_ph(cval_f32_vec, 0);
  const uint16_t cval_fp16 = _mm_extract_epi16(cval_f16_vec, 0);

  /* Process output voxels */
  #pragma omp parallel for collapse(2) schedule(static)
  for (npy_intp oz = 0; oz < OD; ++oz) {
    for (npy_intp oy = 0; oy < OH; ++oy) {
      uint16_t *out_row = out_data + oz * out_stride_z + oy * out_stride_y;

      /* Base source coordinates for this row (SciPy: matrix * [oz, oy, ox]) */
      float base_z = m00 * (float)oz + m01 * (float)oy + shift_z;
      float base_y = m10 * (float)oz + m11 * (float)oy + shift_y;
      float base_x = m20 * (float)oz + m21 * (float)oy + shift_x;

      npy_intp ox = 0;

      /* AVX2 loop - process 8 voxels at a time */
      for (; ox + 7 < OW; ox += 8) {
        /* Generate x indices */
        __m256 vx = _mm256_setr_ps(
            (float)ox, (float)(ox + 1), (float)(ox + 2), (float)(ox + 3),
            (float)(ox + 4), (float)(ox + 5), (float)(ox + 6), (float)(ox + 7));

        /* Compute source coordinates (matrix * [oz, oy, ox]) */
        __m256 zs =
            _mm256_fmadd_ps(_mm256_set1_ps(m02), vx, _mm256_set1_ps(base_z));
        __m256 ys =
            _mm256_fmadd_ps(_mm256_set1_ps(m12), vx, _mm256_set1_ps(base_y));
        __m256 xs =
            _mm256_fmadd_ps(_mm256_set1_ps(m22), vx, _mm256_set1_ps(base_x));

        /* Floor to get integer coordinates */
        __m256 z_floor = _mm256_floor_ps(zs);
        __m256 y_floor = _mm256_floor_ps(ys);
        __m256 x_floor = _mm256_floor_ps(xs);

        /* Get fractional parts */
        __m256 fz = _mm256_sub_ps(zs, z_floor);
        __m256 fy = _mm256_sub_ps(ys, y_floor);
        __m256 fx = _mm256_sub_ps(xs, x_floor);

        /* Compute trilinear weights (8 corners) */
        __m256 one_minus_fx = _mm256_sub_ps(one, fx);
        __m256 one_minus_fy = _mm256_sub_ps(one, fy);
        __m256 one_minus_fz = _mm256_sub_ps(one, fz);

        __m256 w000 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, one_minus_fy),
                                    one_minus_fz);
        __m256 w001 =
            _mm256_mul_ps(_mm256_mul_ps(fx, one_minus_fy), one_minus_fz);
        __m256 w010 =
            _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, fy), one_minus_fz);
        __m256 w011 = _mm256_mul_ps(_mm256_mul_ps(fx, fy), one_minus_fz);
        __m256 w100 =
            _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, one_minus_fy), fz);
        __m256 w101 = _mm256_mul_ps(_mm256_mul_ps(fx, one_minus_fy), fz);
        __m256 w110 = _mm256_mul_ps(_mm256_mul_ps(one_minus_fx, fy), fz);
        __m256 w111 = _mm256_mul_ps(_mm256_mul_ps(fx, fy), fz);

        /* Convert to integers */
        __m256i zi = _mm256_cvttps_epi32(z_floor);
        __m256i yi = _mm256_cvttps_epi32(y_floor);
        __m256i xi = _mm256_cvttps_epi32(x_floor);

        /* Extract values */
        SCIPY_ALIGN(32) int xi_arr[8];
        SCIPY_ALIGN(32) int yi_arr[8];
        SCIPY_ALIGN(32) int zi_arr[8];
        SCIPY_ALIGN(32) float weights[8][8]; /* 8 corners x 8 voxels */

        _mm256_storeu_si256((__m256i *)zi_arr, zi);
        _mm256_storeu_si256((__m256i *)yi_arr, yi);
        _mm256_storeu_si256((__m256i *)xi_arr, xi);

        _mm256_store_ps(weights[0], w000);
        _mm256_store_ps(weights[1], w001);
        _mm256_store_ps(weights[2], w010);
        _mm256_store_ps(weights[3], w011);
        _mm256_store_ps(weights[4], w100);
        _mm256_store_ps(weights[5], w101);
        _mm256_store_ps(weights[6], w110);
        _mm256_store_ps(weights[7], w111);

        SCIPY_ALIGN(32) float result[8];

        /* Process each voxel */
        for (int j = 0; j < 8; j++) {
          int z0 = zi_arr[j];
          int y0 = yi_arr[j];
          int x0 = xi_arr[j];

          /* Check bounds */
          if (x0 >= 0 && x0 < W - 1 && y0 >= 0 && y0 < H - 1 && z0 >= 0 &&
              z0 < D - 1) {

            /* Gather 8 corner FP16 values */
            uint16_t fp16_corners[8];
            npy_intp idx000 =
                z0 * in_stride_z + y0 * in_stride_y + x0 * in_stride_x;
            npy_intp idx001 = idx000 + in_stride_x;
            npy_intp idx010 = idx000 + in_stride_y;
            npy_intp idx011 = idx010 + in_stride_x;
            npy_intp idx100 = idx000 + in_stride_z;
            npy_intp idx101 = idx100 + in_stride_x;
            npy_intp idx110 = idx100 + in_stride_y;
            npy_intp idx111 = idx110 + in_stride_x;

            fp16_corners[0] = in_data[idx000]; /* v000 */
            fp16_corners[1] = in_data[idx001]; /* v001 */
            fp16_corners[2] = in_data[idx010]; /* v010 */
            fp16_corners[3] = in_data[idx011]; /* v011 */
            fp16_corners[4] = in_data[idx100]; /* v100 */
            fp16_corners[5] = in_data[idx101]; /* v101 */
            fp16_corners[6] = in_data[idx110]; /* v110 */
            fp16_corners[7] = in_data[idx111]; /* v111 */

            /* Convert 8 FP16 values to FP32 */
            __m128i fp16_vec = _mm_loadu_si128((__m128i *)fp16_corners);
            __m256 fp32_corners = _mm256_cvtph_ps(fp16_vec);

            SCIPY_ALIGN(32) float corners_f32[8];
            _mm256_store_ps(corners_f32, fp32_corners);

            /* Trilinear interpolation */
            result[j] = corners_f32[0] * weights[0][j] +
                        corners_f32[1] * weights[1][j] +
                        corners_f32[2] * weights[2][j] +
                        corners_f32[3] * weights[3][j] +
                        corners_f32[4] * weights[4][j] +
                        corners_f32[5] * weights[5][j] +
                        corners_f32[6] * weights[6][j] +
                        corners_f32[7] * weights[7][j];
          } else {
            /* Convert cval to float for consistency */
            __m128i cval_vec = _mm_set1_epi16(cval_fp16);
            __m256 cval_f32 = _mm256_cvtph_ps(cval_vec);
            result[j] = _mm256_cvtss_f32(cval_f32);
          }
        }

        /* Convert results from FP32 to FP16 and store */
        __m256 vresult = _mm256_load_ps(result);
        __m128i fp16_result = _mm256_cvtps_ph(vresult, 0);

        /* Vectorized store to output - same optimization as 2D */
        _mm_storeu_si128((__m128i *)(out_row + ox * out_stride_x), fp16_result);
      }

      /* Scalar cleanup */
      for (; ox < OW; ox++) {
        float z_src = m02 * (float)ox + base_z;
        float y_src = m12 * (float)ox + base_y;
        float x_src = m22 * (float)ox + base_x;

        int z0 = (int)floorf(z_src);
        int y0 = (int)floorf(y_src);
        int x0 = (int)floorf(x_src);

        if (x0 >= 0 && x0 < W - 1 && y0 >= 0 && y0 < H - 1 && z0 >= 0 &&
            z0 < D - 1) {

          float fx = x_src - x0;
          float fy = y_src - y0;
          float fz = z_src - z0;

          /* Gather 8 corner FP16 values */
          uint16_t fp16_corners[8];
          npy_intp idx000 =
              z0 * in_stride_z + y0 * in_stride_y + x0 * in_stride_x;
          npy_intp idx001 = idx000 + in_stride_x;
          npy_intp idx010 = idx000 + in_stride_y;
          npy_intp idx011 = idx010 + in_stride_x;
          npy_intp idx100 = idx000 + in_stride_z;
          npy_intp idx101 = idx100 + in_stride_x;
          npy_intp idx110 = idx100 + in_stride_y;
          npy_intp idx111 = idx110 + in_stride_x;

          fp16_corners[0] = in_data[idx000]; /* v000 */
          fp16_corners[1] = in_data[idx001]; /* v001 */
          fp16_corners[2] = in_data[idx010]; /* v010 */
          fp16_corners[3] = in_data[idx011]; /* v011 */
          fp16_corners[4] = in_data[idx100]; /* v100 */
          fp16_corners[5] = in_data[idx101]; /* v101 */
          fp16_corners[6] = in_data[idx110]; /* v110 */
          fp16_corners[7] = in_data[idx111]; /* v111 */

          /* Convert 8 FP16 values to FP32 */
          __m128i fp16_vec = _mm_loadu_si128((__m128i *)fp16_corners);
          __m256 fp32_corners = _mm256_cvtph_ps(fp16_vec);

          SCIPY_ALIGN(32) float corners_f32[8];
          _mm256_store_ps(corners_f32, fp32_corners);

          /* Trilinear interpolation */
          float result_f32 =
              corners_f32[0] * (1 - fx) * (1 - fy) * (1 - fz) + /* v000 */
              corners_f32[1] * fx * (1 - fy) * (1 - fz) +       /* v001 */
              corners_f32[2] * (1 - fx) * fy * (1 - fz) +       /* v010 */
              corners_f32[3] * fx * fy * (1 - fz) +             /* v011 */
              corners_f32[4] * (1 - fx) * (1 - fy) * fz +       /* v100 */
              corners_f32[5] * fx * (1 - fy) * fz +             /* v101 */
              corners_f32[6] * (1 - fx) * fy * fz +             /* v110 */
              corners_f32[7] * fx * fy * fz;                    /* v111 */

          /* Convert result back to FP16 */
          __m256 result_vec = _mm256_set1_ps(result_f32);
          __m128i result_fp16 = _mm256_cvtps_ph(result_vec, 0);
          uint16_t result_val = _mm_extract_epi16(result_fp16, 0);

          out_row[ox * out_stride_x] = result_val;
        } else {
          out_row[ox * out_stride_x] = cval_fp16;
        }
      }
    }
  }

  return 1;
}

#endif /* __AVX2__ */
