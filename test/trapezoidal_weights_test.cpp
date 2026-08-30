#include "trapezoidal_weights.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

bool check_weights(std::span<const double> coordinates, std::span<const double> expected) {
    const std::vector<double> actual = deac_numerics::trapezoidal_weights(coordinates);
    if (actual.size() != expected.size()) {
        std::cerr << "weight count differs: expected " << expected.size()
                  << ", got " << actual.size() << '\n';
        return false;
    }

    for (std::size_t i=0; i<actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::cerr << "weight " << i << " differs: expected " << expected[i]
                      << ", got " << actual[i] << '\n';
            return false;
        }
    }
    return true;
}

bool rejects_too_few_coordinates(std::span<const double> coordinates) {
    try {
        (void) deac_numerics::trapezoidal_weights(coordinates);
    } catch (const std::invalid_argument&) {
        return true;
    }
    std::cerr << "expected fewer than two coordinates to be rejected\n";
    return false;
}

} // namespace

int main() {
    const std::array<double, 2> two_point_grid{1.0, 5.0};
    const std::array<double, 2> two_point_weights{2.0, 2.0};
    if (!check_weights(two_point_grid, two_point_weights)) {
        return 1;
    }

    const std::array<double, 4> nonuniform_grid{0.0, 1.0, 4.0, 10.0};
    const std::array<double, 4> nonuniform_weights{0.5, 2.0, 4.5, 3.0};
    if (!check_weights(nonuniform_grid, nonuniform_weights)) {
        return 1;
    }

    // Repeated coordinates are accepted because solver input validation permits
    // a non-decreasing grid; the helper only applies the integration formula.
    const std::array<double, 3> repeated_grid{0.0, 0.0, 2.0};
    const std::array<double, 3> repeated_weights{0.0, 1.0, 1.0};
    if (!check_weights(repeated_grid, repeated_weights)) {
        return 1;
    }

    const std::array<double, 0> empty_grid{};
    const std::array<double, 1> singleton_grid{0.0};
    if (!rejects_too_few_coordinates(empty_grid)
            || !rejects_too_few_coordinates(singleton_grid)) {
        return 1;
    }

    return 0;
}
