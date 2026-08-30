#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace deac_numerics {

// Return the per-sample weights for trapezoidal integration on an arbitrary
// coordinate grid. Coordinate validation remains the caller's responsibility.
inline std::vector<double> trapezoidal_weights(std::span<const double> coordinates) {
    if (coordinates.size() < 2) {
        throw std::invalid_argument("trapezoidal integration requires at least two coordinates");
    }

    std::vector<double> weights(coordinates.size());
    weights.front() = 0.5*(coordinates[1] - coordinates[0]);
    for (std::size_t i=1; i + 1 < coordinates.size(); ++i) {
        weights[i] = 0.5*(coordinates[i+1] - coordinates[i-1]);
    }
    weights.back() = 0.5*(coordinates.back() - coordinates[coordinates.size()-2]);
    return weights;
}

} // namespace deac_numerics
