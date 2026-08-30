#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#if (defined(__GNUC__) || defined(__clang__)) \
        && defined(__AVX512F__) && !defined(USE_GPU)
    #include <immintrin.h>
    #define DEAC_TRIAL_POPULATION_AVX512 1
#endif

namespace deac_numerics {

#if defined(__GNUC__) || defined(__clang__)
    #define DEAC_NOINLINE __attribute__((noinline))
#else
    #define DEAC_NOINLINE
#endif

// Form one differential-evolution trial row.
//
// Every pointer names a contiguous range of genome_size elements. The output,
// current, three mutant, and mask ranges must be pairwise non-overlapping.
// Solver mutant selection guarantees that the four input population rows are
// distinct, while the trial row and mask live in separate allocations. Making
// that ownership contract explicit lets CPU compilers vectorize the selected
// row operation without changing scalar arithmetic, selected value bits,
// floating-point exception behavior, or RNG order.
template<bool AllowNegativeSpectralWeight>
DEAC_NOINLINE void form_trial_population_row(
        double* __restrict__ trial_row,
        const double* __restrict__ current_row,
        const double* __restrict__ mutant_row1,
        const double* __restrict__ mutant_row2,
        const double* __restrict__ mutant_row3,
        const bool* __restrict__ mutation_mask,
        double differential_weight,
        std::size_t genome_size) {
    std::size_t genome_index = 0;
    #ifdef DEAC_TRIAL_POPULATION_AVX512
        const __m512d differential_weight_vector =
                _mm512_set1_pd(differential_weight);
        const __m512i magnitude_mask = _mm512_set1_epi64(
                static_cast<long long>(UINT64_C(0x7fffffffffffffff)));
        for (; genome_index + 8 <= genome_size; genome_index += 8) {
            // Reading bool values through their declared type preserves the
            // mask representation contract while producing an AVX-512 k-mask.
            const __mmask8 lane_mask = static_cast<__mmask8>(
                    static_cast<unsigned>(mutation_mask[genome_index])
                    | (static_cast<unsigned>(
                            mutation_mask[genome_index + 1]) << 1)
                    | (static_cast<unsigned>(
                            mutation_mask[genome_index + 2]) << 2)
                    | (static_cast<unsigned>(
                            mutation_mask[genome_index + 3]) << 3)
                    | (static_cast<unsigned>(
                            mutation_mask[genome_index + 4]) << 4)
                    | (static_cast<unsigned>(
                            mutation_mask[genome_index + 5]) << 5)
                    | (static_cast<unsigned>(
                            mutation_mask[genome_index + 6]) << 6)
                    | (static_cast<unsigned>(
                            mutation_mask[genome_index + 7]) << 7));

            // Sanitize every arithmetic input before exposing it to the
            // compiler's contraction policy. The boundary makes all lanes of
            // these vectors observable, so inactive values must remain benign
            // zeros even when later arithmetic is emitted without predicates.
            __m512d mutant_values1 = _mm512_maskz_loadu_pd(
                    lane_mask, mutant_row1 + genome_index);
            __m512d mutant_values2 = _mm512_maskz_loadu_pd(
                    lane_mask, mutant_row2 + genome_index);
            __m512d mutant_values3 = _mm512_maskz_loadu_pd(
                    lane_mask, mutant_row3 + genome_index);
            __m512d differential_weights = _mm512_maskz_mov_pd(
                    lane_mask, differential_weight_vector);
            __asm__ volatile(
                    ""
                    : "+v"(mutant_values1),
                      "+v"(mutant_values2),
                      "+v"(mutant_values3),
                      "+v"(differential_weights));

            __m512d evolved_values = mutant_values1
                    + differential_weights*(mutant_values2 - mutant_values3);
            if constexpr (!AllowNegativeSpectralWeight) {
                evolved_values = _mm512_castsi512_pd(_mm512_and_epi64(
                        _mm512_castpd_si512(evolved_values),
                        magnitude_mask));
            }
            const __m512d current_values =
                    _mm512_loadu_pd(current_row + genome_index);
            _mm512_storeu_pd(
                    trial_row + genome_index,
                    _mm512_mask_mov_pd(
                            current_values, lane_mask, evolved_values));
        }
    #endif
    for (; genome_index<genome_size; ++genome_index) {
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

// Two-sided finite-temperature spectra share the evolved zero-frequency value.
// Both rows must contain at least one element and must not overlap.
inline void couple_trial_population_zero(
        double* __restrict__ negative_trial_row,
        const double* __restrict__ positive_trial_row) {
    negative_trial_row[0] = positive_trial_row[0];
}

#undef DEAC_NOINLINE
#ifdef DEAC_TRIAL_POPULATION_AVX512
    #undef DEAC_TRIAL_POPULATION_AVX512
#endif

} // namespace deac_numerics
