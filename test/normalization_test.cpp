#include "normalization.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

bool rejects_target(double target) {
    try {
        deac_numerics::validate_normalization_target(target);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

bool rejects_denominator(double denominator) {
    try {
        (void) deac_numerics::checked_normalization_scale(1.0, denominator);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    const std::array<double, 5> invalid_targets{
            0.0,
            -1.0,
            std::numeric_limits<double>::denorm_min(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN()};
    for (const double target : invalid_targets) {
        if (!rejects_target(target)) {
            std::cerr << "accepted invalid normalization target " << target << '\n';
            return 1;
        }
    }

    const std::array<double, 3> valid_targets{
            std::numeric_limits<double>::min(),
            1.0,
            std::numeric_limits<double>::max()};
    for (const double target : valid_targets) {
        try {
            deac_numerics::validate_normalization_target(target);
        } catch (const std::exception& error) {
            std::cerr << "rejected valid normalization target " << target
                      << ": " << error.what() << '\n';
            return 1;
        }
    }
    try {
        deac_numerics::validate_initial_normalization_scale(1.0, 6.0);
    } catch (const std::exception& error) {
        std::cerr << "rejected a representable initial normalization scale: "
                  << error.what() << '\n';
        return 1;
    }
    try {
        deac_numerics::validate_initial_normalization_scale(
                std::numeric_limits<double>::min(), 6.0);
        std::cerr << "accepted an initial normalization scale that is subnormal\n";
        return 1;
    } catch (const std::invalid_argument&) {
    }
    try {
        deac_numerics::validate_initial_normalization_scale(
                std::numeric_limits<double>::min(),
                std::numeric_limits<double>::denorm_min());
        std::cerr << "accepted a subnormal initial denominator bound\n";
        return 1;
    } catch (const std::invalid_argument&) {
    }

    const std::array<double, 5> invalid_denominators{
            0.0,
            -1.0,
            std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::quiet_NaN()};
    for (const double denominator : invalid_denominators) {
        if (!rejects_denominator(denominator)) {
            std::cerr << "accepted invalid normalization denominator "
                      << denominator << '\n';
            return 1;
        }
    }

    if (deac_numerics::checked_normalization_scale(1.0, 2.0) != 0.5) {
        std::cerr << "changed the valid normalization scale\n";
        return 1;
    }
    if (!rejects_denominator(std::numeric_limits<double>::denorm_min())) {
        std::cerr << "accepted a denominator whose scale overflows\n";
        return 1;
    }
    try {
        (void) deac_numerics::checked_normalization_scale(
                std::numeric_limits<double>::min(),
                std::numeric_limits<double>::denorm_min());
        std::cerr << "accepted a subnormal normalization denominator\n";
        return 1;
    } catch (const std::runtime_error&) {
    }
    try {
        (void) deac_numerics::checked_normalization_scale(
                std::numeric_limits<double>::min(), 2.0);
        std::cerr << "accepted a subnormal normalization scale\n";
        return 1;
    } catch (const std::runtime_error&) {
    }

    std::array<double, 2> valid_candidate{1.0, 3.0};
    if (!deac_numerics::try_apply_normalization(
                2.0, 4.0, valid_candidate)
            || valid_candidate != std::array<double, 2>{0.5, 1.5}) {
        std::cerr << "changed a representable evolved normalization row\n";
        return 1;
    }

    const std::array<double, 2> incumbent{0.5, 1.5};
    std::array<double, 2> degenerate_candidate{0.0, 0.0};
    const bool degenerate_valid = deac_numerics::try_apply_normalization(
            2.0, 0.0, degenerate_candidate);
    const double degenerate_fitness = degenerate_valid
            ? 0.0 : std::numeric_limits<double>::max();
    if (degenerate_valid
            || degenerate_candidate != std::array<double, 2>{0.0, 0.0}
            || degenerate_fitness <= 1.0
            || incumbent != std::array<double, 2>{0.5, 1.5}) {
        std::cerr << "degenerate evolved normalization row was not rejected\n";
        return 1;
    }

    return 0;
}
