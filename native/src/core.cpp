#define PYMIXEF_CORE_BUILD
#include "pymixef_core.h"

#include <cmath>
#include <cstddef>
#include <limits>

extern "C" {

const char* pymixef_core_version(void) {
    return "0.1.1";
}

int pymixef_weighted_sum_squares(
    const size_t n,
    const double* residual,
    const double* weight,
    double* output
) {
    if (residual == nullptr || weight == nullptr || output == nullptr) {
        return PYMIXEF_INVALID_ARGUMENT;
    }
    // Neumaier compensated summation avoids avoidable loss when clusters carry
    // very different scales.
    double sum = 0.0;
    double correction = 0.0;
    for (size_t index = 0; index < n; ++index) {
        if (!std::isfinite(residual[index]) || !std::isfinite(weight[index])
            || weight[index] < 0.0) {
            return PYMIXEF_INVALID_ARGUMENT;
        }
        const double value = weight[index] * residual[index] * residual[index];
        const double updated = sum + value;
        correction += (std::abs(sum) >= std::abs(value))
            ? (sum - updated) + value
            : (value - updated) + sum;
        sum = updated;
    }
    *output = sum + correction;
    return PYMIXEF_OK;
}

int pymixef_cholesky_lower(
    const size_t n,
    const double* matrix,
    double* lower
) {
    if (n == 0 || matrix == nullptr || lower == nullptr) {
        return PYMIXEF_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < n * n; ++index) {
        lower[index] = 0.0;
    }
    for (size_t row = 0; row < n; ++row) {
        for (size_t column = 0; column <= row; ++column) {
            double value = matrix[row * n + column];
            if (!std::isfinite(value)) {
                return PYMIXEF_INVALID_ARGUMENT;
            }
            for (size_t inner = 0; inner < column; ++inner) {
                value -= lower[row * n + inner] * lower[column * n + inner];
            }
            if (row == column) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    return PYMIXEF_NOT_POSITIVE_DEFINITE;
                }
                lower[row * n + column] = std::sqrt(value);
            } else {
                lower[row * n + column] =
                    value / lower[column * n + column];
            }
        }
    }
    return PYMIXEF_OK;
}

}  // extern "C"
