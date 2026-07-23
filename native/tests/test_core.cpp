#include "pymixef_core.h"

#include <cmath>
#include <iostream>
#include <string_view>

int main() {
    if (std::string_view(pymixef_core_version()) != "0.1.1") {
        std::cerr << "unexpected native-core ABI version\n";
        return 1;
    }

    const double residual[] = {1.0, -2.0, 3.0};
    const double weight[] = {1.0, 0.5, 2.0};
    double sum = 0.0;
    if (pymixef_weighted_sum_squares(3, residual, weight, &sum) != PYMIXEF_OK) {
        std::cerr << "weighted sum-of-squares returned an error\n";
        return 2;
    }
    if (std::abs(sum - 21.0) >= 1e-14) {
        std::cerr << "weighted sum-of-squares mismatch\n";
        return 3;
    }

    const double matrix[] = {4.0, 2.0, 2.0, 3.0};
    double lower[4] = {};
    if (pymixef_cholesky_lower(2, matrix, lower) != PYMIXEF_OK) {
        std::cerr << "Cholesky factorization returned an error\n";
        return 4;
    }
    if (
        std::abs(lower[0] - 2.0) >= 1e-14
        || std::abs(lower[2] - 1.0) >= 1e-14
        || std::abs(lower[3] - std::sqrt(2.0)) >= 1e-14
    ) {
        std::cerr << "Cholesky factor mismatch\n";
        return 5;
    }

    const double indefinite[] = {1.0, 2.0, 2.0, 1.0};
    if (
        pymixef_cholesky_lower(2, indefinite, lower)
        != PYMIXEF_NOT_POSITIVE_DEFINITE
    ) {
        std::cerr << "indefinite matrix was not rejected\n";
        return 6;
    }
    return 0;
}
