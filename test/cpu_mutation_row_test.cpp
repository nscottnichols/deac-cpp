#include "cpu_mutation_row.hpp"
#include "trial_population.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace {

using RngWords = std::array<std::uint64_t, 4>;

struct Scenario {
    std::string_view name;
    std::size_t population_size;
    std::size_t genome_size;
    std::uint64_t seed;
    bool inactive_exceptional_inputs;
    bool golden_two_sided_trace;
};

RngWords rng_words(const xoshiro256p_state& rng) {
    return {rng.s[0], rng.s[1], rng.s[2], rng.s[3]};
}

bool same_bits(double left, double right) {
    return std::bit_cast<std::uint64_t>(left)
            == std::bit_cast<std::uint64_t>(right);
}

void reference_select_mutant_indices(
        xoshiro256p_state* rng,
        std::span<std::size_t, 3> mutant_indices,
        std::size_t current_index,
        std::size_t population_size) {
    std::fill(mutant_indices.begin(), mutant_indices.end(), current_index);
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

std::uint64_t reference_probability_threshold(double probability) {
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

void reference_generate_mask(
        xoshiro256p_state* rng,
        bool* mask,
        std::size_t genome_size,
        double crossover_probability) {
    const std::uint64_t threshold =
            reference_probability_threshold(crossover_probability);
    for (std::size_t genome_index=0;
            genome_index<genome_size;
            ++genome_index) {
        mask[genome_index] = (xoshiro256p(rng) >> 11) < threshold;
    }
}

double finite_population_value(
        std::size_t side,
        std::size_t row,
        std::size_t genome_index) {
    const std::size_t encoded =
            (side*53 + row*37 + genome_index*19 + 11)%127;
    const double magnitude = static_cast<double>(encoded + 1)/16.0;
    return (side + row + genome_index)%3 == 0 ? -magnitude : magnitude;
}

void fill_exceptional_population(
        std::vector<double>& population,
        std::size_t side,
        std::size_t population_size,
        std::size_t genome_size) {
    constexpr std::array<std::uint64_t, 9> exceptional_bits{
            UINT64_C(0x8000000000000000), // negative zero
            UINT64_C(0x7ff8000000000042), // quiet NaN with payload
            UINT64_C(0xfff0000000000000), // negative infinity
            UINT64_C(0x0000000000000001), // least positive subnormal
            UINT64_C(0x0000000000000000), // positive zero
            UINT64_C(0x7ff0000000000042), // signaling NaN with payload
            UINT64_C(0x7fefffffffffffff), // greatest finite value
            UINT64_C(0xffefffffffffffff), // least finite value
            UINT64_C(0x8000000000000001)}; // least negative subnormal
    for (std::size_t row=0; row<population_size; ++row) {
        for (std::size_t genome_index=0;
                genome_index<genome_size;
                ++genome_index) {
            const std::size_t bit_index =
                    (side*4 + row + genome_index)%exceptional_bits.size();
            population[row*genome_size + genome_index] =
                    std::bit_cast<double>(exceptional_bits[bit_index]);
        }
    }
}

template<bool AllowNegativeSpectralWeight>
void form_row(
        std::vector<double>& trial_population,
        const std::vector<double>& old_population,
        const std::vector<std::size_t>& mutant_indices,
        const bool* mutation_mask,
        const std::vector<double>& differential_weights,
        std::size_t population_index,
        std::size_t genome_size) {
    const std::size_t mutant1 = mutant_indices[3*population_index];
    const std::size_t mutant2 = mutant_indices[3*population_index + 1];
    const std::size_t mutant3 = mutant_indices[3*population_index + 2];
    deac_numerics::form_trial_population_row<
            AllowNegativeSpectralWeight>(
            trial_population.data() + population_index*genome_size,
            old_population.data() + population_index*genome_size,
            old_population.data() + mutant1*genome_size,
            old_population.data() + mutant2*genome_size,
            old_population.data() + mutant3*genome_size,
            mutation_mask,
            differential_weights[population_index],
            genome_size);
}

bool check_golden_two_sided_trace(
        const Scenario& scenario,
        const std::vector<std::size_t>& mutant_indices,
        const std::vector<RngWords>& states_after_row,
        const xoshiro256p_state& final_rng) {
    if (!scenario.golden_two_sided_trace) {
        return true;
    }
    constexpr std::array<std::size_t, 12> expected_mutants{
            1, 3, 2,
            2, 3, 0,
            1, 3, 0,
            0, 1, 2};
    constexpr std::array<RngWords, 4> expected_states{{
            {UINT64_C(0xf062aab9f7a6faa2), UINT64_C(0x0dcf77b756212491),
             UINT64_C(0x939d6a25fdfe9730), UINT64_C(0x0b644d76635a94c0)},
            {UINT64_C(0x8034594f138ded43), UINT64_C(0xdb242e231e96ee8b),
             UINT64_C(0xe1b40fcbb1195fa5), UINT64_C(0x7a17998cc11cefae)},
            {UINT64_C(0xcec9e4d182a72a2a), UINT64_C(0x931bba2c187363c3),
             UINT64_C(0x425c49474a079085), UINT64_C(0xb2ec70213960e772)},
            {UINT64_C(0x238d8c43697b9ba5), UINT64_C(0x07b9cf401dfbfef5),
             UINT64_C(0xd8da37cc69461b73), UINT64_C(0x462efeb01ad3576c)}}};
    constexpr std::array<std::uint64_t, 4> expected_next{
            UINT64_C(0x69bc8af3844ef311),
            UINT64_C(0x774de5e654913321),
            UINT64_C(0x9ba0be8a414b0c79),
            UINT64_C(0x2db3541122fe2407)};

    if (!std::equal(
                mutant_indices.begin(),
                mutant_indices.end(),
                expected_mutants.begin(),
                expected_mutants.end())
            || !std::equal(
                states_after_row.begin(),
                states_after_row.end(),
                expected_states.begin(),
                expected_states.end())) {
        std::cerr << scenario.name << ": golden mutant/RNG trace mismatch\n";
        return false;
    }
    xoshiro256p_state next_rng = final_rng;
    for (std::size_t index=0; index<expected_next.size(); ++index) {
        const std::uint64_t actual = xoshiro256p(&next_rng);
        if (actual != expected_next[index]) {
            std::cerr << scenario.name << ": golden next RNG mismatch at "
                      << index << ": expected 0x" << std::hex
                      << expected_next[index] << ", got 0x" << actual
                      << std::dec << '\n';
            return false;
        }
    }
    return true;
}

template<bool AllowNegativeSpectralWeight, bool TwoSided>
bool check_scenario(const Scenario& scenario) {
    const std::size_t population_size = scenario.population_size;
    const std::size_t genome_size = scenario.genome_size;
    const std::size_t element_count = population_size*genome_size;

    std::vector<double> positive_old(element_count);
    std::vector<double> negative_old(element_count);
    if (scenario.inactive_exceptional_inputs) {
        fill_exceptional_population(
                positive_old, 0, population_size, genome_size);
        fill_exceptional_population(
                negative_old, 1, population_size, genome_size);
    } else {
        for (std::size_t row=0; row<population_size; ++row) {
            for (std::size_t genome_index=0;
                    genome_index<genome_size;
                    ++genome_index) {
                positive_old[row*genome_size + genome_index] =
                        finite_population_value(0, row, genome_index);
                negative_old[row*genome_size + genome_index] =
                        finite_population_value(1, row, genome_index);
            }
        }
    }

    constexpr std::array<double, 6> probability_cycle{
            0.0,
            1.0,
            0.5,
            0x1.fffffffffffffp-2,
            0x1.0000000000001p-1,
            0.9};
    constexpr std::array<double, 4> weight_cycle{0.0, 0.5, 1.0, 2.0};
    const double signaling_weight = std::bit_cast<double>(
            UINT64_C(0x7ff0000000000043));
    std::vector<double> positive_probabilities(population_size);
    std::vector<double> negative_probabilities(population_size);
    std::vector<double> positive_weights(population_size);
    std::vector<double> negative_weights(population_size);
    for (std::size_t row=0; row<population_size; ++row) {
        positive_probabilities[row] = scenario.inactive_exceptional_inputs
                ? 0.0
                : probability_cycle[row%probability_cycle.size()];
        negative_probabilities[row] = scenario.inactive_exceptional_inputs
                ? 0.0
                : probability_cycle[
                    (probability_cycle.size() - 1
                     - row%probability_cycle.size())];
        positive_weights[row] = scenario.inactive_exceptional_inputs
                ? signaling_weight
                : weight_cycle[row%weight_cycle.size()];
        negative_weights[row] = scenario.inactive_exceptional_inputs
                ? signaling_weight
                : weight_cycle[(row + 1)%weight_cycle.size()];
    }

    std::vector<std::size_t> reference_mutants(3*population_size);
    std::vector<std::size_t> actual_mutants(3*population_size);
    auto reference_positive_masks = std::make_unique<bool[]>(element_count);
    auto actual_positive_masks = std::make_unique<bool[]>(element_count);
    std::unique_ptr<bool[]> reference_negative_masks;
    std::unique_ptr<bool[]> actual_negative_masks;
    if constexpr (TwoSided) {
        reference_negative_masks = std::make_unique<bool[]>(element_count);
        actual_negative_masks = std::make_unique<bool[]>(element_count);
    }
    std::vector<double> reference_positive_trial(element_count, -901.0);
    std::vector<double> actual_positive_trial(element_count, -902.0);
    std::vector<double> reference_negative_trial(element_count, -903.0);
    std::vector<double> actual_negative_trial(element_count, -904.0);
    std::vector<RngWords> reference_states;
    std::vector<RngWords> actual_states;
    reference_states.reserve(population_size);
    actual_states.reserve(population_size);

    xoshiro256p_state reference_rng = xoshiro256p_init(scenario.seed);
    std::feclearexcept(FE_ALL_EXCEPT);
    for (std::size_t row=0; row<population_size; ++row) {
        reference_select_mutant_indices(
                &reference_rng,
                std::span<std::size_t, 3>(
                    reference_mutants.data() + 3*row, 3),
                row,
                population_size);
        reference_generate_mask(
                &reference_rng,
                reference_positive_masks.get() + row*genome_size,
                genome_size,
                positive_probabilities[row]);
        if constexpr (TwoSided) {
            reference_generate_mask(
                    &reference_rng,
                    reference_negative_masks.get() + row*genome_size,
                    genome_size,
                    negative_probabilities[row]);
        }
        reference_states.push_back(rng_words(reference_rng));
    }
    for (std::size_t row=0; row<population_size; ++row) {
        form_row<AllowNegativeSpectralWeight>(
                reference_positive_trial,
                positive_old,
                reference_mutants,
                reference_positive_masks.get() + row*genome_size,
                positive_weights,
                row,
                genome_size);
        if constexpr (TwoSided) {
            form_row<AllowNegativeSpectralWeight>(
                    reference_negative_trial,
                    negative_old,
                    reference_mutants,
                    reference_negative_masks.get() + row*genome_size,
                    negative_weights,
                    row,
                    genome_size);
            deac_numerics::couple_trial_population_zero(
                    reference_negative_trial.data() + row*genome_size,
                    reference_positive_trial.data() + row*genome_size);
        }
    }
    const int reference_exceptions = std::fetestexcept(FE_ALL_EXCEPT);

    xoshiro256p_state actual_rng = xoshiro256p_init(scenario.seed);
    auto positive_scratch = std::make_unique<bool[]>(genome_size);
    std::unique_ptr<bool[]> negative_scratch;
    if constexpr (TwoSided) {
        negative_scratch = std::make_unique<bool[]>(genome_size);
    }
    std::feclearexcept(FE_ALL_EXCEPT);
    for (std::size_t row=0; row<population_size; ++row) {
        std::fill_n(positive_scratch.get(), genome_size, row%2 == 0);
        const std::span<std::size_t, 3> row_mutants(
                actual_mutants.data() + 3*row, 3);
        const std::span<bool> positive_mask(
                positive_scratch.get(), genome_size);
        if constexpr (TwoSided) {
            std::fill_n(negative_scratch.get(), genome_size, row%2 != 0);
            const std::span<bool> negative_mask(
                    negative_scratch.get(), genome_size);
            deac_numerics::generate_cpu_mutation_row_inputs(
                    &actual_rng,
                    row,
                    population_size,
                    row_mutants,
                    positive_mask,
                    positive_probabilities[row],
                    negative_mask,
                    negative_probabilities[row]);
            std::copy_n(
                    negative_scratch.get(),
                    genome_size,
                    actual_negative_masks.get() + row*genome_size);
        } else {
            deac_numerics::generate_cpu_mutation_row_inputs(
                    &actual_rng,
                    row,
                    population_size,
                    row_mutants,
                    positive_mask,
                    positive_probabilities[row]);
        }
        std::copy_n(
                positive_scratch.get(),
                genome_size,
                actual_positive_masks.get() + row*genome_size);

        form_row<AllowNegativeSpectralWeight>(
                actual_positive_trial,
                positive_old,
                actual_mutants,
                positive_scratch.get(),
                positive_weights,
                row,
                genome_size);
        if constexpr (TwoSided) {
            form_row<AllowNegativeSpectralWeight>(
                    actual_negative_trial,
                    negative_old,
                    actual_mutants,
                    negative_scratch.get(),
                    negative_weights,
                    row,
                    genome_size);
            deac_numerics::couple_trial_population_zero(
                    actual_negative_trial.data() + row*genome_size,
                    actual_positive_trial.data() + row*genome_size);
        }
        actual_states.push_back(rng_words(actual_rng));
    }
    const int actual_exceptions = std::fetestexcept(FE_ALL_EXCEPT);

    const auto fail = [&](std::string_view what) {
        std::cerr << scenario.name
                  << " failed: " << what
                  << ", population=" << population_size
                  << ", genome=" << genome_size
                  << ", signed=" << AllowNegativeSpectralWeight
                  << ", two-sided=" << TwoSided << '\n';
        return false;
    };

    if (reference_mutants != actual_mutants) {
        return fail("mutant indices differ");
    }
    for (std::size_t row=0; row<population_size; ++row) {
        for (std::size_t slot=0; slot<3; ++slot) {
            const std::size_t mutant = actual_mutants[3*row + slot];
            if (mutant >= population_size || mutant == row) {
                return fail("invalid mutant index");
            }
            for (std::size_t previous=0; previous<slot; ++previous) {
                if (mutant == actual_mutants[3*row + previous]) {
                    return fail("duplicate mutant index");
                }
            }
        }
    }
    if (reference_states != actual_states
            || rng_words(reference_rng) != rng_words(actual_rng)) {
        return fail("RNG state differs");
    }
    xoshiro256p_state reference_next = reference_rng;
    xoshiro256p_state actual_next = actual_rng;
    for (std::size_t index=0; index<8; ++index) {
        if (xoshiro256p(&reference_next) != xoshiro256p(&actual_next)) {
            return fail("next RNG output differs");
        }
    }
    if (rng_words(reference_next) != rng_words(actual_next)) {
        return fail("RNG state after next-output probe differs");
    }
    for (std::size_t index=0; index<element_count; ++index) {
        if (reference_positive_masks[index] != actual_positive_masks[index]) {
            return fail("positive mutation mask differs");
        }
        if constexpr (TwoSided) {
            if (reference_negative_masks[index]
                    != actual_negative_masks[index]) {
                return fail("negative mutation mask differs");
            }
        }
        if (!same_bits(
                    reference_positive_trial[index],
                    actual_positive_trial[index])) {
            return fail("positive trial bits differ");
        }
        if constexpr (TwoSided) {
            if (!same_bits(
                        reference_negative_trial[index],
                        actual_negative_trial[index])) {
                return fail("negative trial bits differ");
            }
        }
    }
    for (std::size_t row=0; row<population_size; ++row) {
        for (std::size_t genome_index=0;
                genome_index<genome_size;
                ++genome_index) {
            const std::size_t index = row*genome_size + genome_index;
            if (positive_probabilities[row] == 0.0
                    && actual_positive_masks[index]) {
                return fail("zero-probability positive mask is true");
            }
            if (positive_probabilities[row] == 1.0
                    && !actual_positive_masks[index]) {
                return fail("unit-probability positive mask is false");
            }
            if constexpr (TwoSided) {
                if (negative_probabilities[row] == 0.0
                        && actual_negative_masks[index]) {
                    return fail("zero-probability negative mask is true");
                }
                if (negative_probabilities[row] == 1.0
                        && !actual_negative_masks[index]) {
                    return fail("unit-probability negative mask is false");
                }
            }
        }
        if constexpr (TwoSided) {
            const std::size_t zero_index = row*genome_size;
            if (!same_bits(
                        actual_negative_trial[zero_index],
                        actual_positive_trial[zero_index])) {
                return fail("two-sided zero-frequency coupling differs");
            }
        }
    }
    if (reference_exceptions != actual_exceptions) {
        return fail("floating-point exception flags differ");
    }
    if (scenario.inactive_exceptional_inputs) {
        if (reference_exceptions != 0 || actual_exceptions != 0) {
            return fail("inactive exceptional inputs raised an exception");
        }
        for (std::size_t row=0; row<population_size; ++row) {
            for (std::size_t genome_index=0;
                    genome_index<genome_size;
                    ++genome_index) {
                const std::size_t index = row*genome_size + genome_index;
                if (!same_bits(actual_positive_trial[index], positive_old[index])) {
                    return fail("inactive positive lane changed bits");
                }
                if constexpr (TwoSided) {
                    const double expected = genome_index == 0
                            ? positive_old[row*genome_size]
                            : negative_old[index];
                    if (!same_bits(actual_negative_trial[index], expected)) {
                        return fail("inactive negative lane/coupling changed bits");
                    }
                }
            }
        }
    }
    if constexpr (TwoSided) {
        if (!check_golden_two_sided_trace(
                    scenario, actual_mutants, actual_states, actual_rng)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    constexpr std::array<Scenario, 12> scenarios{{
            {"single-element-mask", 4, 1, UINT64_C(0xdecafbad), false, false},
            {"two-element-mask", 7, 2, UINT64_C(47), false, false},
            {"short-tail", 5, 7, UINT64_C(17), false, false},
            {"one-vector", 5, 8, UINT64_C(8675309), false, false},
            {"golden-vector-tail", 4, 9, UINT64_C(123), false, true},
            {"fifteen-lanes", 9, 15, UINT64_C(23), false, false},
            {"sixteen-lanes", 15, 16, UINT64_C(0), false, false},
            {"two-vectors-tail", 8, 17, UINT64_C(1), false, false},
            {"production-width-minus-one", 16, 1023, UINT64_C(17), false, false},
            {"production-width", 8, 1024, UINT64_C(123), false, false},
            {"production-width-plus-one", 17, 1025, UINT64_C(47), false, false},
            {"inactive-exceptional", 4, 9, UINT64_C(123), true, false}}};

    for (const Scenario& scenario : scenarios) {
        if (!check_scenario<false, false>(scenario)
                || !check_scenario<true, false>(scenario)
                || !check_scenario<false, true>(scenario)
                || !check_scenario<true, true>(scenario)) {
            return 1;
        }
    }
    return 0;
}
