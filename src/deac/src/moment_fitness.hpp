#pragma once

#include <cmath>
#include <cstddef>
#include <span>

namespace deac_numerics {

inline double scalar_chi_square_penalty(
        double calculated, double observed, double standard_deviation) {
    const double term = (observed - calculated)/standard_deviation;
    return term*term;
}

struct ObjectiveMomentView {
    bool active = false;
    const double* calculated = nullptr;
    double observed = 0.0;
    double standard_deviation = 1.0;
};

// Score one population row without changing the solver's historical
// arithmetic order.  The residual is accumulated in timeslice order and
// divided only after the complete sum.  Moment penalties are then added in
// negative-first, first, and third order.  Inactive moment pointers are never
// evaluated.
inline double score_population_objective_row(
        std::span<const double> observed_isf,
        std::span<const double> calculated_isf_row,
        std::span<const double> isf_error,
        std::size_t population_index,
        const ObjectiveMomentView& negative_first_moment,
        const ObjectiveMomentView& first_moment,
        const ObjectiveMomentView& third_moment) {
    double fitness = 0.0;
    for (std::size_t index=0; index<observed_isf.size(); ++index) {
        fitness += std::pow(
                (observed_isf[index] - calculated_isf_row[index])
                    /isf_error[index],
                2);
    }
    fitness /= observed_isf.size();

    if (negative_first_moment.active) {
        fitness += std::pow(
                (negative_first_moment.observed
                    - negative_first_moment.calculated[population_index])
                    /negative_first_moment.standard_deviation,
                2);
    }
    if (first_moment.active) {
        fitness += scalar_chi_square_penalty(
                first_moment.calculated[population_index],
                first_moment.observed,
                first_moment.standard_deviation);
    }
    if (third_moment.active) {
        fitness += std::pow(
                (third_moment.observed
                    - third_moment.calculated[population_index])
                    /third_moment.standard_deviation,
                2);
    }
    return fitness;
}

} // namespace deac_numerics
