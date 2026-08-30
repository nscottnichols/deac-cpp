#include "evolution_controls.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using deac_configuration::EvolutionControls;
using deac_configuration::validate_evolution_controls;

bool accepts(const EvolutionControls& controls, const char* description) {
    try {
        validate_evolution_controls(controls);
    } catch (const std::invalid_argument& error) {
        std::cerr << "expected " << description << " to be accepted, got: "
                  << error.what() << '\n';
        return false;
    }
    return true;
}

bool rejects(
        const EvolutionControls& controls,
        const char* expected_parameter,
        const char* description) {
    try {
        validate_evolution_controls(controls);
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find(expected_parameter) == std::string::npos) {
            std::cerr << "expected " << description << " to identify "
                      << expected_parameter << ", got: " << error.what() << '\n';
            return false;
        }
        return true;
    }
    std::cerr << "expected " << description << " to be rejected\n";
    return false;
}

} // namespace

int main() {
    const EvolutionControls defaults{0.9, 0.1, 0.9, 0.1, 1.0};
    const EvolutionControls lower_boundaries{0.0, 0.0, 0.0, 0.0, -42.0};
    const EvolutionControls upper_boundaries{1.0, 1.0, 2.0, 1.0, 42.0};
    if (!accepts(defaults, "default controls")
            || !accepts(lower_boundaries, "lower boundaries and negative stop threshold")
            || !accepts(upper_boundaries, "upper boundaries")) {
        return 1;
    }

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();

    EvolutionControls controls = defaults;
    controls.crossover_probability = -0.01;
    if (!rejects(controls, "crossover_probability", "negative crossover probability")) return 1;
    controls.crossover_probability = 1.01;
    if (!rejects(controls, "crossover_probability", "crossover probability above one")) return 1;
    controls.crossover_probability = nan;
    if (!rejects(controls, "crossover_probability", "non-finite crossover probability")) return 1;

    controls = defaults;
    controls.self_adapting_crossover_probability = -0.01;
    if (!rejects(controls, "self_adapting_crossover_probability", "negative crossover adaptation probability")) return 1;
    controls.self_adapting_crossover_probability = 1.01;
    if (!rejects(controls, "self_adapting_crossover_probability", "crossover adaptation probability above one")) return 1;
    controls.self_adapting_crossover_probability = infinity;
    if (!rejects(controls, "self_adapting_crossover_probability", "non-finite crossover adaptation probability")) return 1;

    controls = defaults;
    controls.differential_weight = -0.01;
    if (!rejects(controls, "differential_weight", "negative differential weight")) return 1;
    controls.differential_weight = 2.01;
    if (!rejects(controls, "differential_weight", "differential weight above two")) return 1;
    controls.differential_weight = -infinity;
    if (!rejects(controls, "differential_weight", "non-finite differential weight")) return 1;

    controls = defaults;
    controls.self_adapting_differential_weight_probability = -0.01;
    if (!rejects(controls, "self_adapting_differential_weight_probability", "negative differential-weight adaptation probability")) return 1;
    controls.self_adapting_differential_weight_probability = 1.01;
    if (!rejects(controls, "self_adapting_differential_weight_probability", "differential-weight adaptation probability above one")) return 1;
    controls.self_adapting_differential_weight_probability = nan;
    if (!rejects(controls, "self_adapting_differential_weight_probability", "non-finite differential-weight adaptation probability")) return 1;

    controls = defaults;
    controls.stop_minimum_fitness = nan;
    if (!rejects(controls, "stop_minimum_fitness", "NaN stop threshold")) return 1;
    controls.stop_minimum_fitness = infinity;
    if (!rejects(controls, "stop_minimum_fitness", "infinite stop threshold")) return 1;
    controls.stop_minimum_fitness = -infinity;
    if (!rejects(controls, "stop_minimum_fitness", "negative-infinite stop threshold")) return 1;

    return 0;
}
