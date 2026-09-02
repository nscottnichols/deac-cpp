#include "moment_fitness.hpp"
#include "population_scoring.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <span>

namespace {

constexpr std::size_t population_size = 2;
constexpr std::size_t genome_size = 3;
constexpr std::size_t number_of_timeslices = 2;

using Population = std::array<double, population_size*genome_size>;
using ModeledIsf = std::array<double, population_size*number_of_timeslices>;
using Moment = std::array<double, population_size>;

bool check_projection(
        const char* description,
        const deac_numerics::PopulationView& population,
        const double* negative_isf_kernel,
        const double* negative_first_terms,
        const double* negative_third_terms,
        const ModeledIsf& expected_isf,
        const Moment& expected_first,
        const Moment& expected_third,
        const Moment& expected_negative_first,
        const Moment& expected_fitness) {
    const std::array<double, number_of_timeslices*genome_size>
            positive_isf_kernel{1.0, 2.0, 4.0, 0.5, 1.0, 2.0};
    const std::array<double, genome_size> positive_first_terms{0.0, 1.0, 2.0};
    const std::array<double, genome_size> positive_third_terms{0.0, 1.0, 4.0};
    const std::array<double, number_of_timeslices> negative_first_isf_terms{1.0, 2.0};

    ModeledIsf modeled_isf{};
    Moment first_moment{};
    Moment third_moment{};
    Moment negative_first_moment{};
    modeled_isf.fill(101.0);
    first_moment.fill(102.0);
    third_moment.fill(103.0);
    negative_first_moment.fill(104.0);
    deac_numerics::project_population_forward_model(
            modeled_isf.data(),
            population,
            positive_isf_kernel.data(),
            negative_isf_kernel,
            number_of_timeslices,
            genome_size,
            population_size);
    deac_numerics::project_population_scalar_moment(
            first_moment.data(),
            population,
            positive_first_terms.data(),
            negative_first_terms,
            genome_size,
            population_size);
    deac_numerics::project_population_scalar_moment(
            third_moment.data(),
            population,
            positive_third_terms.data(),
            negative_third_terms,
            genome_size,
            population_size);
    deac_numerics::project_modeled_isf_moment(
            negative_first_moment.data(),
            modeled_isf.data(),
            negative_first_isf_terms.data(),
            number_of_timeslices,
            population_size);

    if (modeled_isf != expected_isf
            || first_moment != expected_first
            || third_moment != expected_third
            || negative_first_moment != expected_negative_first) {
        std::cerr << description << " projection changed\n";
        return false;
    }

    const std::array<double, number_of_timeslices> observed_isf{1.0, 2.0};
    const std::array<double, number_of_timeslices> isf_error{1.0, 0.5};
    const deac_numerics::ObjectiveMomentView negative_first_objective{
            true, negative_first_moment.data(), 0.0, 2.0};
    const deac_numerics::ObjectiveMomentView first_objective{
            true, first_moment.data(), 0.0, 0.5};
    const deac_numerics::ObjectiveMomentView third_objective{
            true, third_moment.data(), 0.0, 4.0};
    Moment fitness{};
    for (std::size_t population_index=0;
            population_index<population_size;
            ++population_index) {
        fitness[population_index] =
                deac_numerics::score_population_objective_row(
                        observed_isf,
                        std::span<const double>(
                                modeled_isf.data()
                                    + population_index*number_of_timeslices,
                                number_of_timeslices),
                        isf_error,
                        population_index,
                        negative_first_objective,
                        first_objective,
                        third_objective);
    }
    if (fitness != expected_fitness) {
        std::cerr << description << " objective changed\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    Population incumbent_positive{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Population incumbent_negative{1.0, 7.0, 8.0, 4.0, 9.0, 10.0};
    Population candidate_positive = incumbent_positive;
    Population candidate_negative = incumbent_negative;
    const Population original_incumbent_positive = incumbent_positive;
    const Population original_incumbent_negative = incumbent_negative;
    const Population original_candidate_positive = candidate_positive;
    const Population original_candidate_negative = candidate_negative;

    if (incumbent_positive.data() == candidate_positive.data()
            || incumbent_negative.data() == candidate_negative.data()) {
        std::cerr << "incumbent and candidate fixtures alias\n";
        return 1;
    }

    const std::array<double, number_of_timeslices*genome_size>
            negative_isf_kernel{8.0, 4.0, 2.0, 4.0, 2.0, 1.0};
    const std::array<double, genome_size> negative_first_terms{0.0, -1.0, -2.0};
    const std::array<double, genome_size> negative_third_terms{0.0, -1.0, -4.0};

    const ModeledIsf one_sided_isf{17.0, 8.5, 38.0, 19.0};
    const Moment one_sided_first{8.0, 17.0};
    const Moment one_sided_third{14.0, 29.0};
    const Moment one_sided_negative_first{34.0, 76.0};
    const Moment one_sided_fitness{769.75, 3915.0625};
    const ModeledIsf two_sided_isf{69.0, 34.5, 126.0, 63.0};
    const Moment two_sided_first{-15.0, -12.0};
    const Moment two_sided_third{-25.0, -20.0};
    const Moment two_sided_negative_first{138.0, 252.0};
    const Moment two_sided_fitness{10124.5625, 31731.5};

    const auto check_both_views = [&](
            const char* one_sided_description,
            const char* two_sided_description,
            Population& positive,
            Population& negative) {
        return check_projection(
                       one_sided_description,
                       {positive.data(), nullptr},
                       nullptr,
                       nullptr,
                       nullptr,
                       one_sided_isf,
                       one_sided_first,
                       one_sided_third,
                       one_sided_negative_first,
                       one_sided_fitness)
                && check_projection(
                       two_sided_description,
                       {positive.data(), negative.data()},
                       negative_isf_kernel.data(),
                       negative_first_terms.data(),
                       negative_third_terms.data(),
                       two_sided_isf,
                       two_sided_first,
                       two_sided_third,
                       two_sided_negative_first,
                       two_sided_fitness);
    };

    if (!check_both_views(
                "one-sided incumbent", "two-sided incumbent",
                incumbent_positive, incumbent_negative)
            || !check_both_views(
                "one-sided candidate", "two-sided candidate",
                candidate_positive, candidate_negative)) {
        return 1;
    }

    if (incumbent_positive != original_incumbent_positive
            || incumbent_negative != original_incumbent_negative
            || candidate_positive != original_candidate_positive
            || candidate_negative != original_candidate_negative) {
        std::cerr << "population projections modified their source arrays\n";
        return 1;
    }
    return 0;
}
