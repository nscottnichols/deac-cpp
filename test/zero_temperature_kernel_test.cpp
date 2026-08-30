#include "zero_temperature_kernel.hpp"

#include <cmath>

namespace {

bool check_close(double actual, double expected) {
    return std::abs(actual - expected) <= 1e-14;
}

}  // namespace

int main() {
    using deac_numerics::zero_temperature_laplace_term;
    using deac_numerics::zero_temperature_third_moment_term;

    if (!check_close(zero_temperature_laplace_term(0.0, 3.0, 0.25), 0.25)
            || !check_close(zero_temperature_laplace_term(2.0, 0.0, 0.75), 0.75)
            || !check_close(
                    zero_temperature_laplace_term(0.5, 2.0, 1.5),
                    1.5*std::exp(-1.0))
            || !check_close(
                    zero_temperature_third_moment_term(2.0, 1.5), 12.0)) {
        return 1;
    }

    // A small nonuniform-grid forward projection protects both the one-sided
    // Laplace factors and the caller-owned quadrature weights.
    const double spectrum[] = {2.0, 3.0, 5.0};
    const double frequency[] = {0.0, 1.0, 3.0};
    const double weights[] = {0.25, 1.5, 0.75};
    constexpr double tau = 0.4;
    double projection = 0.0;
    for (int index = 0; index < 3; ++index) {
        projection += spectrum[index]*zero_temperature_laplace_term(
                tau, frequency[index], weights[index]);
    }
    const double expected =
            2.0*0.25 + 3.0*1.5*std::exp(-0.4) + 5.0*0.75*std::exp(-1.2);
    return check_close(projection, expected) ? 0 : 1;
}
