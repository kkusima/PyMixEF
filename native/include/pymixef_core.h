#ifndef PYMIXEF_CORE_H
#define PYMIXEF_CORE_H

#include <stddef.h>

#if defined(_WIN32)
#  if defined(PYMIXEF_CORE_BUILD)
#    define PYMIXEF_API __declspec(dllexport)
#  else
#    define PYMIXEF_API __declspec(dllimport)
#  endif
#else
#  define PYMIXEF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum pymixef_status {
    PYMIXEF_OK = 0,
    PYMIXEF_INVALID_ARGUMENT = 1,
    PYMIXEF_NOT_POSITIVE_DEFINITE = 2
};

PYMIXEF_API const char* pymixef_core_version(void);

PYMIXEF_API int pymixef_weighted_sum_squares(
    size_t n,
    const double* residual,
    const double* weight,
    double* output
);

/*
 * Factor a symmetric positive-definite row-major matrix. Only the lower
 * triangle of `lower` is populated; callers must allocate n*n doubles.
 */
PYMIXEF_API int pymixef_cholesky_lower(
    size_t n,
    const double* matrix,
    double* lower
);

#ifdef __cplusplus
}
#endif

#endif

