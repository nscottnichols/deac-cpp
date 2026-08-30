#pragma once

namespace deac_numerics {

inline double scalar_chi_square_penalty(
        double calculated, double observed, double standard_deviation) {
    const double term = (observed - calculated)/standard_deviation;
    return term*term;
}

} // namespace deac_numerics
