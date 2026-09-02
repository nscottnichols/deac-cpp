#include "moment_fitness.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>

namespace {

using deac_numerics::ObjectiveMomentView;

double score(
        std::span<const double> observed,
        std::span<const double> calculated,
        std::span<const double> error,
        std::size_t population_index,
        const ObjectiveMomentView& negative_first = {},
        const ObjectiveMomentView& first = {},
        const ObjectiveMomentView& third = {}) {
    return deac_numerics::score_population_objective_row(
            observed, calculated, error, population_index,
            negative_first, first, third);
}

} // namespace

int main() {
    if (deac_numerics::scalar_chi_square_penalty(2.0, 0.0, 1.0) != 4.0) {
        std::cerr << "zero-target first-moment penalty is not finite and squared\n";
        return 1;
    }
    if (deac_numerics::scalar_chi_square_penalty(2.0, 1.0, 0.5) != 4.0) {
        std::cerr << "scalar penalty does not divide by the error before squaring\n";
        return 1;
    }
    if (!std::isfinite(
                deac_numerics::scalar_chi_square_penalty(0.0, 0.0, 1.0))) {
        std::cerr << "matching zero moment produced a non-finite penalty\n";
        return 1;
    }

    const std::array<double, 2> observed_isf{1.0, 4.0};
    const std::array<double, 2> calculated_isf{0.0, 2.0};
    const std::array<double, 2> isf_error{1.0, 2.0};
    if (score(observed_isf, calculated_isf, isf_error, 0) != 1.0) {
        std::cerr << "residual-only objective changed\n";
        return 1;
    }

    // Index zero is deliberately different so the test also proves that each
    // active observable selects the requested population row.
    const std::array<double, 2> negative_first_calculated{99.0, 4.0};
    const std::array<double, 2> first_calculated{99.0, 1.0};
    const std::array<double, 2> third_calculated{99.0, 5.0};
    const ObjectiveMomentView negative_first{
            true, negative_first_calculated.data(), 2.0, 2.0};
    const ObjectiveMomentView first{
            true, first_calculated.data(), 0.0, 0.5};
    const ObjectiveMomentView third{
            true, third_calculated.data(), 1.0, 2.0};
    const std::array<ObjectiveMomentView, 3> moments{
            negative_first, first, third};
    const std::array<double, 3> penalties{1.0, 4.0, 4.0};

    for (unsigned int active_mask=0; active_mask<8; ++active_mask) {
        std::array<ObjectiveMomentView, 3> selected = moments;
        double expected = 1.0;
        for (std::size_t index=0; index<selected.size(); ++index) {
            selected[index].active = (active_mask & (1U << index)) != 0;
            if (selected[index].active) {
                expected += penalties[index];
            }
        }
        const double actual = score(
                observed_isf, calculated_isf, isf_error, 1,
                selected[0], selected[1], selected[2]);
        if (actual != expected) {
            std::cerr << "active moment combination " << active_mask
                      << " produced " << actual
                      << ", expected " << expected << '\n';
            return 1;
        }
    }

    // The negative-first penalty is 2^54, whose ULP is four.  The production
    // order first loses the residual in that large addition and then loses the
    // unit first- and third-moment penalties independently.  Regrouping the
    // three small contributions first changes the result by one ULP.
    const std::array<double, 1> rounding_observed{1.0};
    const std::array<double, 1> rounding_calculated{0.0};
    const std::array<double, 1> rounding_error{1.0};
    const std::array<double, 1> large_moment{std::ldexp(1.0, 27)};
    const std::array<double, 1> unit_moment{1.0};
    const ObjectiveMomentView large_penalty{
            true, large_moment.data(), 0.0, 1.0};
    const ObjectiveMomentView unit_penalty{
            true, unit_moment.data(), 0.0, 1.0};
    const double ordered = score(
            rounding_observed, rounding_calculated, rounding_error, 0,
            large_penalty, unit_penalty, unit_penalty);
    const double large_penalty_value = std::ldexp(1.0, 54);
    const double regrouped = large_penalty_value + (1.0 + 1.0 + 1.0);
    if (ordered != large_penalty_value || regrouped == ordered) {
        std::cerr << "rounding-sensitive objective addition order changed\n";
        return 1;
    }
    return 0;
}
