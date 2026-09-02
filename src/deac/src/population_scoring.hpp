#pragma once

#include "population_projection.hpp"

#include <algorithm>
#include <cstddef>

namespace deac_numerics {

// Opaque non-owning view whose pointers retain the active backend's existing
// layout: row-major on CPU and column-major on accelerators.  Only the CPU
// projection helpers below interpret the pointed-to storage.  The
// negative-frequency population is absent for one-sided configurations.
struct PopulationView {
    double* positive_frequency = nullptr;
    double* negative_frequency = nullptr;
};

// Reset and project a population through the forward-model kernels.  Keep the
// historical arithmetic order: every positive-frequency contribution is
// accumulated before any negative-frequency contribution, and the underlying
// projection evaluates kernel*population for each genome entry.
inline void project_population_forward_model(
        double* modeled_isf,
        const PopulationView& population,
        const double* positive_frequency_kernel,
        const double* negative_frequency_kernel,
        std::size_t number_of_timeslices,
        std::size_t genome_size,
        std::size_t population_size) {
    std::fill(
            modeled_isf,
            modeled_isf + population_size*number_of_timeslices,
            0.0);
    accumulate_population_projection(
            modeled_isf,
            positive_frequency_kernel,
            population.positive_frequency,
            number_of_timeslices,
            genome_size,
            population_size);
    if (population.negative_frequency != nullptr) {
        accumulate_population_projection(
                modeled_isf,
                negative_frequency_kernel,
                population.negative_frequency,
                number_of_timeslices,
                genome_size,
                population_size);
    }
}

inline void accumulate_population_scalar_projection(
        double* projected,
        const double* population,
        const double* terms,
        std::size_t genome_size,
        std::size_t population_size) {
    for (std::size_t population_index=0;
            population_index<population_size;
            ++population_index) {
        for (std::size_t genome_index=0;
                genome_index<genome_size;
                ++genome_index) {
            projected[population_index] +=
                    population[population_index*genome_size + genome_index]
                    *terms[genome_index];
        }
    }
}

// Reset and project a scalar moment from a population.  The positive and
// optional negative terms use the same population*term operand order as the
// solver's original CPU matrix-vector helper.
inline void project_population_scalar_moment(
        double* projected_moment,
        const PopulationView& population,
        const double* positive_frequency_terms,
        const double* negative_frequency_terms,
        std::size_t genome_size,
        std::size_t population_size) {
    std::fill(
            projected_moment,
            projected_moment + population_size,
            0.0);
    accumulate_population_scalar_projection(
            projected_moment,
            population.positive_frequency,
            positive_frequency_terms,
            genome_size,
            population_size);
    if (population.negative_frequency != nullptr) {
        accumulate_population_scalar_projection(
                projected_moment,
                population.negative_frequency,
                negative_frequency_terms,
                genome_size,
                population_size);
    }
}

// Reset and project the negative-first observable from an already modeled ISF.
// Modeled ISF rows are population-major, matching the forward-model output.
inline void project_modeled_isf_moment(
        double* projected_moment,
        const double* modeled_isf,
        const double* terms,
        std::size_t number_of_timeslices,
        std::size_t population_size) {
    std::fill(
            projected_moment,
            projected_moment + population_size,
            0.0);
    accumulate_population_scalar_projection(
            projected_moment,
            modeled_isf,
            terms,
            number_of_timeslices,
            population_size);
}

} // namespace deac_numerics
