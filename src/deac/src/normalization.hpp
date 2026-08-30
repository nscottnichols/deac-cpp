#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace deac_numerics {

inline void validate_normalization_target(double target) {
    if (!std::isfinite(target)
            || target < std::numeric_limits<double>::min()) {
        throw std::invalid_argument(
                "--normalize requires the first ISF value (zeroth moment) "
                "to be finite, positive, and at least the smallest normal double");
    }
}

struct NormalizationTerms {
    std::vector<double> positive_frequency;
    std::vector<double> negative_frequency;
    double maximum_initial_denominator = 0.0;
};

inline NormalizationTerms make_normalization_terms(
        std::span<const double> frequency,
        std::span<const double> frequency_weights,
        double temperature) {
    NormalizationTerms terms;
    terms.positive_frequency.resize(frequency.size());
    #if !defined(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF) && !defined(ZEROT)
        terms.negative_frequency.resize(frequency.size());
    #endif

    #ifndef ZEROT
        [[maybe_unused]] const double beta = 1.0/temperature;
    #else
        (void) temperature;
    #endif

    for (std::size_t index=0; index<frequency.size(); ++index) {
        [[maybe_unused]] const double value = frequency[index];
        const double weight = frequency_weights[index];
        double positive_term;
        #if !defined(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF) && !defined(ZEROT)
            double negative_term;
        #endif

        #ifndef ZEROT
            #ifdef USE_HYPERBOLIC_MODEL
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    positive_term = weight*std::cosh(0.5*beta*value);
                #else
                    positive_term = 0.5*weight/std::exp(-0.5*beta*value);
                    negative_term = 0.5*weight*std::exp(-0.5*beta*value);
                #endif
            #endif
            #ifdef USE_STANDARD_MODEL
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    positive_term = weight*(1.0 + std::exp(-beta*value));
                #else
                    positive_term = weight;
                    negative_term = weight;
                #endif
            #endif
            #ifdef USE_NORMALIZATION_MODEL
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    positive_term = weight;
                #else
                    positive_term = weight*(1.0/(1.0 + std::exp(-beta*value)));
                    const double e_to_bf = std::exp(-beta*value);
                    negative_term = weight*(e_to_bf/(1.0 + e_to_bf));
                #endif
            #endif
        #else
            positive_term = weight;
        #endif

        if (!std::isfinite(positive_term) || positive_term < 0.0) {
            throw std::invalid_argument(
                    "normalization weights must be finite and non-negative");
        }
        terms.positive_frequency[index] = positive_term;
        terms.maximum_initial_denominator += positive_term;

        #if !defined(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF) && !defined(ZEROT)
            if (!std::isfinite(negative_term) || negative_term < 0.0) {
                throw std::invalid_argument(
                        "normalization weights must be finite and non-negative");
            }
            terms.negative_frequency[index] = negative_term;
            terms.maximum_initial_denominator += negative_term;
        #endif
        if (!std::isfinite(terms.maximum_initial_denominator)) {
            throw std::invalid_argument(
                    "normalization weight sum must be finite");
        }
    }

    if (terms.maximum_initial_denominator <= 0.0) {
        throw std::invalid_argument(
                "normalization weight sum must be positive");
    }
    return terms;
}

inline void validate_initial_normalization_scale(
        double target, double maximum_initial_denominator) {
    validate_normalization_target(target);
    const double minimum_scale = target/maximum_initial_denominator;
    if (!std::isnormal(maximum_initial_denominator)
            || !std::isnormal(minimum_scale)) {
        throw std::invalid_argument(
                "--normalize target is outside the representable normal range "
                "for this model and frequency grid");
    }
}

inline double checked_normalization_scale(double target, double denominator) {
    validate_normalization_target(target);
    if (!std::isnormal(denominator) || denominator <= 0.0) {
        throw std::runtime_error(
                "population normalization denominator must be finite, positive, "
                "and normal");
    }

    const double scale = target/denominator;
    if (!std::isnormal(scale)) {
        throw std::runtime_error(
                "population normalization scale must be finite, positive, and normal");
    }
    return scale;
}

inline bool try_apply_normalization(
        double target,
        double denominator,
        std::span<double> positive_frequency,
        std::span<double> negative_frequency = {}) {
    if (!std::isnormal(denominator) || denominator <= 0.0) {
        std::fill(positive_frequency.begin(), positive_frequency.end(), 0.0);
        std::fill(negative_frequency.begin(), negative_frequency.end(), 0.0);
        return false;
    }

    const double scale = target/denominator;
    if (!std::isnormal(scale) || scale <= 0.0) {
        std::fill(positive_frequency.begin(), positive_frequency.end(), 0.0);
        std::fill(negative_frequency.begin(), negative_frequency.end(), 0.0);
        return false;
    }

    bool valid = true;
    for (double& value : positive_frequency) {
        value *= scale;
        valid = valid && std::isfinite(value);
    }
    for (double& value : negative_frequency) {
        value *= scale;
        valid = valid && std::isfinite(value);
    }
    if (!valid) {
        std::fill(positive_frequency.begin(), positive_frequency.end(), 0.0);
        std::fill(negative_frequency.begin(), negative_frequency.end(), 0.0);
    }
    return valid;
}

} // namespace deac_numerics
