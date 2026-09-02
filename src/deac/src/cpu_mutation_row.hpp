#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include <rng.hpp>

namespace deac_numerics {
namespace cpu_mutation_detail {

inline void select_mutant_indices(
        xoshiro256p_state* rng,
        std::span<std::size_t, 3> mutant_indices,
        std::size_t current_index,
        std::size_t population_size) {
    mutant_indices[0] = current_index;
    mutant_indices[1] = current_index;
    mutant_indices[2] = current_index;
    while (mutant_indices[0] == current_index) {
        mutant_indices[0] = xoshiro256p(rng)%population_size;
    }
    while (mutant_indices[1] == current_index
            || mutant_indices[1] == mutant_indices[0]) {
        mutant_indices[1] = xoshiro256p(rng)%population_size;
    }
    while (mutant_indices[2] == current_index
            || mutant_indices[2] == mutant_indices[0]
            || mutant_indices[2] == mutant_indices[1]) {
        mutant_indices[2] = xoshiro256p(rng)%population_size;
    }
}

inline std::uint64_t crossover_probability_threshold(double probability) {
    // The RNG produces q * 2^-53 for an integer q in [0, 2^53).  Comparing q
    // against ceil(probability * 2^53) is exact and avoids converting every
    // generated value to double.
    constexpr std::uint64_t random_range = std::uint64_t{1} << 53;
    if (!(probability > 0.0)) {
        return 0;
    }
    if (probability >= 1.0) {
        return random_range;
    }
    return static_cast<std::uint64_t>(
            std::ceil(std::ldexp(probability, 53)));
}

inline void generate_mutation_mask(
        xoshiro256p_state* rng,
        std::span<bool> mutation_mask,
        double crossover_probability) {
    const std::uint64_t threshold =
            crossover_probability_threshold(crossover_probability);
    for (bool& mutate : mutation_mask) {
        // Draw unconditionally at probability endpoints to retain the exact
        // historical RNG sequence.
        mutate = (xoshiro256p(rng) >> 11) < threshold;
    }
}

} // namespace cpu_mutation_detail

// Generate the complete random input for one CPU trial-population row.  Trial
// arithmetic remains in form_trial_population_row() and introduces no RNG
// calls.  Every span must refer to distinct, writable storage, population_size
// must be at least four, and population_index must be less than it.  In the
// two-sided overload both mask spans must have the same extent.
inline void generate_cpu_mutation_row_inputs(
        xoshiro256p_state* rng,
        std::size_t population_index,
        std::size_t population_size,
        std::span<std::size_t, 3> mutant_indices,
        std::span<bool> positive_mask,
        double positive_crossover_probability) {
    cpu_mutation_detail::select_mutant_indices(
            rng,
            mutant_indices,
            population_index,
            population_size);
    cpu_mutation_detail::generate_mutation_mask(
            rng,
            positive_mask,
            positive_crossover_probability);
}

inline void generate_cpu_mutation_row_inputs(
        xoshiro256p_state* rng,
        std::size_t population_index,
        std::size_t population_size,
        std::span<std::size_t, 3> mutant_indices,
        std::span<bool> positive_mask,
        double positive_crossover_probability,
        std::span<bool> negative_mask,
        double negative_crossover_probability) {
    generate_cpu_mutation_row_inputs(
            rng,
            population_index,
            population_size,
            mutant_indices,
            positive_mask,
            positive_crossover_probability);
    cpu_mutation_detail::generate_mutation_mask(
            rng,
            negative_mask,
            negative_crossover_probability);
}

} // namespace deac_numerics
