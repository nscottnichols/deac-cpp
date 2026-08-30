#include "trial_population.hpp"

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
#include <vector>

namespace {

enum class MaskPattern {
    all_false,
    all_true,
    alternating,
    seeded_random,
};

template<bool AllowNegativeSpectralWeight>
void reference_form_trial_population_row(
        double* trial_row,
        const double* current_row,
        const double* mutant_row1,
        const double* mutant_row2,
        const double* mutant_row3,
        const bool* mutation_mask,
        double differential_weight,
        std::size_t genome_size) {
    for (std::size_t genome_index=0;
            genome_index<genome_size;
            ++genome_index) {
        if (mutation_mask[genome_index]) {
            const double mutant_value =
                    mutant_row1[genome_index]
                    + differential_weight*(
                            mutant_row2[genome_index]
                            - mutant_row3[genome_index]);
            if constexpr (AllowNegativeSpectralWeight) {
                trial_row[genome_index] = mutant_value;
            } else {
                trial_row[genome_index] = std::fabs(mutant_value);
            }
        } else {
            trial_row[genome_index] = current_row[genome_index];
        }
    }
}

double row_value(std::size_t row, std::size_t genome_index) {
    const std::size_t encoded = (row*37 + genome_index*19 + 11)%127;
    const double magnitude = static_cast<double>(encoded + 1)/16.0;
    return (row + genome_index)%3 == 0 ? -magnitude : magnitude;
}

void fill_masks(
        bool* positive_mask,
        bool* negative_mask,
        MaskPattern pattern,
        std::size_t genome_size) {
    std::uint64_t state = UINT64_C(0x6a09e667f3bcc909);
    for (std::size_t genome_index=0;
            genome_index<genome_size;
            ++genome_index) {
        bool mutate = false;
        switch (pattern) {
            case MaskPattern::all_false:
                mutate = false;
                break;
            case MaskPattern::all_true:
                mutate = true;
                break;
            case MaskPattern::alternating:
                mutate = genome_index%2 == 0;
                break;
            case MaskPattern::seeded_random:
                state = state*UINT64_C(6364136223846793005) + UINT64_C(1);
                mutate = (state >> 63) != 0;
                break;
        }
        positive_mask[genome_index] = mutate;
        negative_mask[genome_index] = !mutate;
    }
}

bool same_bits(double left, double right) {
    return std::bit_cast<std::uint64_t>(left)
            == std::bit_cast<std::uint64_t>(right);
}

bool check_inactive_lane_bits() {
    constexpr std::array<std::uint64_t, 9> current_bits{
            UINT64_C(0x8000000000000000), // negative zero
            UINT64_C(0x7ff8000000000042), // quiet NaN with payload
            UINT64_C(0xfff0000000000000), // negative infinity
            UINT64_C(0x0000000000000001), // least positive subnormal
            UINT64_C(0x0000000000000000), // positive zero
            UINT64_C(0x7ff0000000000042), // signaling NaN with payload
            UINT64_C(0x7fefffffffffffff), // greatest finite value
            UINT64_C(0xffefffffffffffff), // least finite value
            UINT64_C(0x8000000000000001)}; // least negative subnormal
    std::array<double, current_bits.size()> current{};
    std::array<double, current_bits.size()> mutant1{};
    std::array<double, current_bits.size()> mutant2{};
    std::array<double, current_bits.size()> mutant3{};
    std::array<double, current_bits.size()> actual{};
    std::array<bool, current_bits.size()> mask{};

    for (std::size_t index=0; index<current.size(); ++index) {
        current[index] = std::bit_cast<double>(current_bits[index]);
        mutant1[index] = std::numeric_limits<double>::max();
        mutant2[index] = std::numeric_limits<double>::infinity();
        mutant3[index] = std::numeric_limits<double>::infinity();
    }

    std::feclearexcept(FE_ALL_EXCEPT);
    deac_numerics::form_trial_population_row<false>(
            actual.data(),
            current.data(),
            mutant1.data(),
            mutant2.data(),
            mutant3.data(),
            mask.data(),
            2.0,
            actual.size());

    if (std::fetestexcept(FE_ALL_EXCEPT) != 0) {
        std::cerr << "inactive trial lanes raised a floating-point exception\n";
        return false;
    }

    for (std::size_t index=0; index<actual.size(); ++index) {
        if (std::bit_cast<std::uint64_t>(actual[index])
                != current_bits[index]) {
            std::cerr << "inactive trial lane changed bits at index=" << index
                      << ": expected 0x" << std::hex << current_bits[index]
                      << ", got 0x"
                      << std::bit_cast<std::uint64_t>(actual[index]) << '\n';
            return false;
        }
    }
    return true;
}

template<bool AllowNegativeSpectralWeight>
bool check_case(
        std::size_t genome_size,
        MaskPattern pattern,
        double positive_weight,
        double negative_weight) {
    std::array<std::vector<double>, 4> positive_rows;
    std::array<std::vector<double>, 4> negative_rows;
    for (std::size_t row=0; row<positive_rows.size(); ++row) {
        positive_rows[row].resize(genome_size);
        negative_rows[row].resize(genome_size);
        for (std::size_t genome_index=0;
                genome_index<genome_size;
                ++genome_index) {
            positive_rows[row][genome_index] = row_value(row, genome_index);
            negative_rows[row][genome_index] =
                    row_value(row + positive_rows.size(), genome_index);
        }
    }

    const std::unique_ptr<bool[]> positive_mask(new bool[genome_size]);
    const std::unique_ptr<bool[]> negative_mask(new bool[genome_size]);
    fill_masks(
            positive_mask.get(),
            negative_mask.get(),
            pattern,
            genome_size);

    std::vector<double> expected_positive(genome_size, -901.0);
    std::vector<double> actual_positive(genome_size, -902.0);
    std::vector<double> expected_negative(genome_size, -903.0);
    std::vector<double> actual_negative(genome_size, -904.0);

    reference_form_trial_population_row<AllowNegativeSpectralWeight>(
            expected_positive.data(),
            positive_rows[0].data(),
            positive_rows[1].data(),
            positive_rows[2].data(),
            positive_rows[3].data(),
            positive_mask.get(),
            positive_weight,
            genome_size);
    reference_form_trial_population_row<AllowNegativeSpectralWeight>(
            expected_negative.data(),
            negative_rows[0].data(),
            negative_rows[1].data(),
            negative_rows[2].data(),
            negative_rows[3].data(),
            negative_mask.get(),
            negative_weight,
            genome_size);
    expected_negative[0] = expected_positive[0];

    deac_numerics::form_trial_population_row<AllowNegativeSpectralWeight>(
            actual_positive.data(),
            positive_rows[0].data(),
            positive_rows[1].data(),
            positive_rows[2].data(),
            positive_rows[3].data(),
            positive_mask.get(),
            positive_weight,
            genome_size);
    deac_numerics::form_trial_population_row<AllowNegativeSpectralWeight>(
            actual_negative.data(),
            negative_rows[0].data(),
            negative_rows[1].data(),
            negative_rows[2].data(),
            negative_rows[3].data(),
            negative_mask.get(),
            negative_weight,
            genome_size);
    deac_numerics::couple_trial_population_zero(
            actual_negative.data(), actual_positive.data());

    for (std::size_t index=0; index<genome_size; ++index) {
        if (!same_bits(expected_positive[index], actual_positive[index])
                || !same_bits(expected_negative[index], actual_negative[index])) {
            std::cerr << "trial row mismatch for width=" << genome_size
                      << ", mask=" << static_cast<int>(pattern)
                      << ", positive F=" << positive_weight
                      << ", negative F=" << negative_weight
                      << ", signed=" << AllowNegativeSpectralWeight
                      << ", index=" << index
                      << ": expected positive " << std::hexfloat
                      << expected_positive[index] << ", got "
                      << actual_positive[index] << "; expected negative "
                      << expected_negative[index] << ", got "
                      << actual_negative[index] << '\n';
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    constexpr std::array<std::size_t, 9> genome_sizes{
            1, 2, 7, 8, 9, 15, 16, 17, 1024};
    constexpr std::array<MaskPattern, 4> patterns{
            MaskPattern::all_false,
            MaskPattern::all_true,
            MaskPattern::alternating,
            MaskPattern::seeded_random};
    constexpr std::array<double, 3> differential_weights{0.0, 0.5, 2.0};

    if (!check_inactive_lane_bits()) {
        return 1;
    }

    for (const std::size_t genome_size : genome_sizes) {
        for (const MaskPattern pattern : patterns) {
            for (const double positive_weight : differential_weights) {
                for (const double negative_weight : differential_weights) {
                    if (!check_case<false>(
                                genome_size,
                                pattern,
                                positive_weight,
                                negative_weight)
                            || !check_case<true>(
                                genome_size,
                                pattern,
                                positive_weight,
                                negative_weight)) {
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}
