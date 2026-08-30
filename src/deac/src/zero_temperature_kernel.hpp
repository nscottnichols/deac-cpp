#pragma once

#include <cmath>

namespace deac_numerics {

// Zero-temperature DSF forward kernel on the non-negative frequency grid:
//     F(tau) = integral_0^infinity exp(-tau*omega) S(omega) d omega.
// The caller supplies the frequency quadrature weight.
inline double zero_temperature_laplace_term(
        double tau, double frequency, double quadrature_weight) noexcept {
    return quadrature_weight*std::exp(-tau*frequency);
}

inline double zero_temperature_third_moment_term(
        double frequency, double quadrature_weight) noexcept {
    return quadrature_weight*frequency*frequency*frequency;
}

}  // namespace deac_numerics
