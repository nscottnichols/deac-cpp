#pragma once

#include <cmath>
#include <stdexcept>

namespace deac_configuration {

struct EvolutionControls {
    double crossover_probability;
    double self_adapting_crossover_probability;
    double differential_weight;
    double self_adapting_differential_weight_probability;
    double stop_minimum_fitness;
};

inline void require_finite_closed_interval(
        double value, double lower, double upper, const char* error_message) {
    if (!std::isfinite(value) || value < lower || value > upper) {
        throw std::invalid_argument(error_message);
    }
}

inline void validate_evolution_controls(const EvolutionControls& controls) {
    require_finite_closed_interval(
            controls.crossover_probability, 0.0, 1.0,
            "crossover_probability must be finite and in [0, 1]");
    require_finite_closed_interval(
            controls.self_adapting_crossover_probability, 0.0, 1.0,
            "self_adapting_crossover_probability must be finite and in [0, 1]");
    require_finite_closed_interval(
            controls.differential_weight, 0.0, 2.0,
            "differential_weight must be finite and in [0, 2]");
    require_finite_closed_interval(
            controls.self_adapting_differential_weight_probability, 0.0, 1.0,
            "self_adapting_differential_weight_probability must be finite and in [0, 1]");

    // A negative threshold is useful when callers want every configured
    // generation to run; only non-finite values are invalid.
    if (!std::isfinite(controls.stop_minimum_fitness)) {
        throw std::invalid_argument("stop_minimum_fitness must be finite");
    }
}

} // namespace deac_configuration
