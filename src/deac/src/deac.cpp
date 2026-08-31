#include <stdio.h>
#include <math.h> // cosh
#include <stdlib.h>
#include <cstdlib>
#include <iostream>
#include <tuple> // for tie() and tuple
#include <argparse.hpp>
#include <sstream> // 
#include <algorithm> // std::none_of
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <span>
#include <vector>
#include <rng.hpp>
#include "evolution_controls.hpp"
#include "moment_fitness.hpp"
#include "normalization.hpp"
#include "population_projection.hpp"
#include "result_io.hpp"
#include "trial_population.hpp"
#include "zero_temperature_kernel.hpp"
#include "trapezoidal_weights.hpp"
#include "build_identity.hpp"
#include <memory> // string_format
#include <string> // string_format
#include <stdexcept> // throw
#include <cassert>
#include <fs.h> //fs namespace (std::filesystem or std::experimental::filesystem)

//GPU acceleration
#ifdef USE_HIP
    #include "deac_gpu.hip.hpp"
#endif
#ifdef USE_CUDA
    #include "deac_gpu.cuh"
#endif
#ifdef USE_SYCL
    #include "deac_gpu.sycl.h"
#endif

// Finite-temperature spectra without detailed balance have independently
// evolved positive- and negative-frequency populations.  ZeroT deliberately
// retains the original one-sided, non-negative-frequency population.
#if !defined(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF) && !defined(ZEROT)
    #define DEAC_TWO_SIDED_POPULATION 1
#endif

[[noreturn]] void fail_with_error(const std::string& error_message) {
    throw std::runtime_error(error_message);
}

#ifdef DEAC_TEST_POISON_GPU_FITNESS
    #define DEAC_TEST_GPU_CALL(expression) GPU_ASSERT(expression)
#endif

void print_build_identity(std::ostream& output) {
    output << deac_build_identity::canonical_json() << std::endl;
}

void print_build_receipt(std::ostream& output) {
    output << deac_build_identity::build_receipt_json() << std::endl;
}

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args ) {
    //See https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
    int size_s = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>( size );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

void matrix_multiply_MxN_by_Nx1(double * C, double * A, double * B, size_t M, size_t N) {
    for (size_t i=0; i<M; i++) {
        for (size_t j=0; j<N; j++) {
            C[i] += A[i*N + j]*B[j];
        }
    }
}

double reduced_chi_square_statistic(double * observed, double * calculated, double * error, size_t length) {
    double chi_squared = 0.0;
    for (size_t i=0; i<length; i++) {
        chi_squared += pow((observed[i] - calculated[i])/error[i],2);
    }
    return chi_squared;
}

double minimum(double * A, size_t length) {
    double _minimum = A[0];
    for (size_t i=0; i<length; i++) {
        if (A[i] < _minimum) {
            _minimum = A[i];
        }
    }
    return _minimum;
}

std::tuple <size_t, double> argmin_and_min(double * A, size_t length) {
    size_t _argmin=0;
    double _minimum = A[0];
    for (size_t i=0; i<length; i++) {
        if (A[i] < _minimum) {
            _minimum = A[i];
            _argmin = i;
        }
    }
    std::tuple <size_t, double> argmin_tuple(_argmin, _minimum);
    return argmin_tuple;
}

double mean(double * A, size_t length) {
    double _mean = 0.0;
    for (size_t i=0; i<length; i++) {
        _mean += A[i];
    }
    return _mean/length;
}

double squared_mean(double * A, size_t length) {
    double _squared_mean = 0.0;
    for (size_t i=0; i<length; i++) {
        _squared_mean += A[i]*A[i];
    }
    return _squared_mean/length;
}

void set_mutant_indices(struct xoshiro256p_state* rng, size_t* mutant_indices, size_t mutant_index0, size_t length) {
    mutant_indices[0] = mutant_index0;
    mutant_indices[1] = mutant_index0;
    mutant_indices[2] = mutant_index0;
    while (mutant_indices[0] == mutant_index0) {
        mutant_indices[0] = xoshiro256p(rng) % length;
    }

    while ((mutant_indices[1] == mutant_index0) || (mutant_indices[1] == mutant_indices[0])) {
        mutant_indices[1] = xoshiro256p(rng) % length;
    }

    while ((mutant_indices[2] == mutant_index0) || (mutant_indices[2] == mutant_indices[0])
            || (mutant_indices[2] == mutant_indices[1])) {
        mutant_indices[2] = xoshiro256p(rng) % length;
    }
}

#ifndef USE_GPU
static uint64_t probability_threshold(double probability) {
    // The RNG produces q * 2^-53 for an integer q in [0, 2^53).
    // Comparing q against ceil(probability * 2^53) is therefore exact and
    // avoids converting every generated value to double.
    constexpr uint64_t random_range = UINT64_C(1) << 53;
    if (!(probability > 0.0)) {
        return 0;
    }
    if (probability >= 1.0) {
        return random_range;
    }
    return static_cast<uint64_t>(std::ceil(std::ldexp(probability, 53)));
}
#endif

void deac(struct xoshiro256p_state * rng, double * const imaginary_time,
        double * const isf, double * const isf_error, double * frequency,
        double temperature, size_t number_of_generations, size_t number_of_timeslices, size_t population_size,
        size_t genome_size, bool normalize, bool use_negative_first_moment, 
        double first_moment, double first_moment_error,
        double third_moment, double third_moment_error,
        double crossover_probability,
        double self_adapting_crossover_probability,
        double differential_weight, 
        double self_adapting_differential_weight_probability,
        double stop_minimum_fitness,
        bool track_stats, size_t seed, std::string uuid_str, std::string spectra_type, fs::path save_directory) {

    #ifdef ZEROT
        //Set flags and temperature
        if (use_negative_first_moment) {
            std::cout << "use_negative_first_moment disabled for zero temperature build" << std::endl;
        }
        use_negative_first_moment = false; //FIXME disabling inverse first moment for zero temperature (needs further investigation)
        temperature = 0.0;
    #else
        #ifdef DEAC_TWO_SIDED_POPULATION
            //Use beta periodicity in imaginary time to reduce numerical instabilities
            double periodicity = 1.0; // Periodic for bosnonic systems
            if (
                (spectra_type == "spfsf") ||
                (spectra_type == "ffull")
               ) {
               periodicity = -1.0; // Antiperiodic for fermionic systems
            }
        #endif
    #endif

    #ifdef USE_GPU
        //Create GPU device streams
        deac_stream_t stream_array[MAX_GPU_STREAMS];
        for (size_t i = 0; i < MAX_GPU_STREAMS; i++) {
            GPU_ASSERT(deac_stream_create(stream_array[i]));
        }

        // Set up default "stream"
        auto default_stream = stream_array[0];
        
        #ifdef USE_BLAS
            //Create GPU device BLAS handles
            deac_blas_handle_t blas_handle_array[MAX_GPU_STREAMS];
            for (size_t i = 0; i < MAX_GPU_STREAMS; i++) {
                GPU_BLAS_ASSERT(deac_create_blas_handle(blas_handle_array[i]));
                GPU_BLAS_ASSERT(deac_set_stream(blas_handle_array[i], stream_array[i]));
            }

            // Set up default BLAS_HANDLE
            auto default_blas_handle = blas_handle_array[0];
        #endif


        #ifdef USE_SYCL
            //Test for valid subgroup size
            auto sg_sizes = default_stream.get_device().get_info<sycl::info::device::sub_group_sizes>();
            if (std::none_of(sg_sizes.cbegin(), sg_sizes.cend(), [](auto i) { return i  == SUB_GROUP_SIZE; })) {
                std::stringstream ss;
                ss << "Invalid SUB_GROUP_SIZE. Please select from: ";
                for (auto it = sg_sizes.cbegin(); it != sg_sizes.cend(); it++) {
                    if (it != sg_sizes.begin()) {
                        ss << " ";
                    }
                    ss << *it;
                }
                throw std::runtime_error(ss.str());
            }
        #endif
    #endif

    #ifdef USE_GPU
        //Load isf and isf error onto GPU
        double* d_isf;
        double* d_isf_error;
        size_t bytes_isf = sizeof(double)*number_of_timeslices;
        size_t bytes_isf_error = sizeof(double)*number_of_timeslices;
        GPU_ASSERT(deac_malloc_device(double, d_isf,       number_of_timeslices, default_stream));
        GPU_ASSERT(deac_malloc_device(double, d_isf_error, number_of_timeslices, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_isf, isf, bytes_isf, default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_isf_error, isf_error, bytes_isf_error, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    #ifndef ZEROT
        double beta = 1.0/temperature;
    #endif
    double zeroth_moment = isf[0];
    bool use_first_moment = first_moment >= 0.0;
    bool use_third_moment = third_moment >= 0.0;
    const std::vector<double> frequency_weights = deac_numerics::trapezoidal_weights(
            std::span<const double>(frequency, genome_size));

    //Set isf term for trapezoidal rule integration with dsf (population members)
    size_t bytes_isf_term = sizeof(double)*genome_size*number_of_timeslices;
    double * isf_term_positive_frequency;
    isf_term_positive_frequency = (double*) malloc(bytes_isf_term);
    #ifdef DEAC_TWO_SIDED_POPULATION
        double * isf_term_negative_frequency;
        isf_term_negative_frequency = (double*) malloc(bytes_isf_term);
    #endif
    for (size_t i=0; i<number_of_timeslices; i++) {
        double t = imaginary_time[i];
        #ifndef ZEROT
            #ifdef USE_HYPERBOLIC_MODEL
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    double bo2mt  = 0.5*beta - t;
                #else
                    double bo2mtp = -0.5*beta - t; // add beta to tau
                    double bo2mtn = 1.5*beta - t; // subtract beta from tau
                #endif
            #endif
            #ifdef USE_STANDARD_MODEL
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    double bmt = beta - t;
                #else
                    double tmb = t - beta; // subtract beta from tau
                #endif
            #endif
            #ifdef USE_NORMALIZATION_MODEL
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    double bmt = beta - t;
                #else
                    double tmb = t - beta; // subtract beta from tau
                #endif
            #endif
        #endif
        for (size_t j=0; j<genome_size; j++) {
            double f = frequency[j];
            (void) f;
            const double df = frequency_weights[j];
            size_t isf_term_idx = i*genome_size + j;
            #ifndef ZEROT
                #ifdef USE_HYPERBOLIC_MODEL
                    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        isf_term_positive_frequency[isf_term_idx] = df*cosh(bo2mt*f);
                    #else
                        isf_term_positive_frequency[isf_term_idx] = periodicity*df*exp(bo2mtp*f)/2;
                        isf_term_negative_frequency[isf_term_idx] = periodicity*df*exp(-bo2mtn*f)/2;
                    #endif
                #endif
                #ifdef USE_STANDARD_MODEL
                    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        isf_term_positive_frequency[isf_term_idx] = df*(exp(-bmt*f) + exp(-t*f));
                    #else
                        isf_term_positive_frequency[isf_term_idx] = df*exp(-t*f);
                        isf_term_negative_frequency[isf_term_idx] = periodicity*df*exp(tmb*f);
                    #endif
                #endif
                #ifdef USE_NORMALIZATION_MODEL
                    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        double _num = exp(-bmt*f) + exp(-t*f);
                        double _denom = 1.0 + exp(-beta*f);
                        isf_term_positive_frequency[isf_term_idx] = df*(_num/_denom);
                    #else
                        // exp(-t*f)/(1 + exp(-beta*f)) == exp(-t*f) - exp(-t*f)/(1 + exp(beta*f))
                        double _num = exp(-t*f);
                        double _denom = 1.0 + exp(-beta*f);
                        isf_term_positive_frequency[isf_term_idx] = df*(_num/_denom);
                        double e_to_mtw_n = exp(tmb*f); // exp(-t*f) with t - beta
                        double _denom_n = 1.0 + exp(-beta*f); // 1.0 + exp(beta*f) where f is negative here
                        isf_term_negative_frequency[isf_term_idx] = periodicity*df*(e_to_mtw_n - e_to_mtw_n/_denom_n);
                    #endif
                #endif
            #else
                isf_term_positive_frequency[isf_term_idx] =
                        deac_numerics::zero_temperature_laplace_term(t, f, df);
            #endif
        }
    }
    
    #ifdef USE_GPU
        //Load isf term onto GPU
        double* d_isf_term_positive_frequency;
        GPU_ASSERT(deac_malloc_device(double, d_isf_term_positive_frequency, genome_size*number_of_timeslices, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_isf_term_positive_frequency, isf_term_positive_frequency, bytes_isf_term, default_stream));
        #ifdef DEAC_TWO_SIDED_POPULATION
            double* d_isf_term_negative_frequency;
            GPU_ASSERT(deac_malloc_device(double, d_isf_term_negative_frequency, genome_size*number_of_timeslices, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_isf_term_negative_frequency, isf_term_negative_frequency, bytes_isf_term, default_stream));
        #endif
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    //Generate population and set initial fitness
    size_t bytes_population = sizeof(double)*genome_size*population_size;
    double * population_old_positive_frequency;
    double * population_new_positive_frequency;
    population_new_positive_frequency = (double*) malloc(bytes_population);
    population_old_positive_frequency = (double*) malloc(bytes_population);
    #ifdef DEAC_TWO_SIDED_POPULATION
        double * population_old_negative_frequency;
        double * population_new_negative_frequency;
        population_old_negative_frequency = (double*) malloc(bytes_population);
        population_new_negative_frequency = (double*) malloc(bytes_population);
    #endif
    for (size_t i=0; i<population_size; i++) {
        for (size_t j=0; j<genome_size; j++) {
            #ifdef USE_GPU
                size_t population_idx = j*population_size + i;
            #else
                size_t population_idx = i*genome_size + j;
            #endif
            population_old_positive_frequency[population_idx] = (xoshiro256p(rng) >> 11) * 0x1.0p-53; // to_double2
            #ifdef ALLOW_NEGATIVE_SPECTRAL_WEIGHT
                population_old_positive_frequency[population_idx] -= 0.5;
            #endif
            #ifdef DEAC_TWO_SIDED_POPULATION
                population_old_negative_frequency[population_idx] = (xoshiro256p(rng) >> 11) * 0x1.0p-53; // to_double2
                #ifdef ALLOW_NEGATIVE_SPECTRAL_WEIGHT
                    population_old_negative_frequency[population_idx] -= 0.5;
                #endif
            #endif
        }
    }
    #ifdef DEAC_TWO_SIDED_POPULATION
        for (size_t i=0; i<population_size; i++) {
            #ifdef USE_GPU
                size_t zero_frequency_idx = i;
            #else
                size_t zero_frequency_idx = i*genome_size;
            #endif
            population_old_negative_frequency[zero_frequency_idx] = population_old_positive_frequency[zero_frequency_idx]; // Match up zero (always take value from positive result)
        }
    #endif

    #ifdef USE_GPU
        double* d_population_old_positive_frequency;
        double* d_population_new_positive_frequency;
        GPU_ASSERT(deac_malloc_device(double, d_population_old_positive_frequency, genome_size*population_size, default_stream));
        GPU_ASSERT(deac_malloc_device(double, d_population_new_positive_frequency, genome_size*population_size, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_population_old_positive_frequency, population_old_positive_frequency, bytes_population, default_stream));
        #ifdef DEAC_TWO_SIDED_POPULATION
            double* d_population_old_negative_frequency;
            double* d_population_new_negative_frequency;
            GPU_ASSERT(deac_malloc_device(double, d_population_old_negative_frequency, genome_size*population_size, default_stream));
            GPU_ASSERT(deac_malloc_device(double, d_population_new_negative_frequency, genome_size*population_size, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_population_old_negative_frequency, population_old_negative_frequency, bytes_population, default_stream));
        #endif
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    // Normalize population
    size_t bytes_normalization = sizeof(double)*population_size;
    size_t bytes_normalization_valid = sizeof(int)*population_size;
    size_t bytes_normalization_term = sizeof(double)*genome_size;
    double * normalization = nullptr;
    int * normalization_valid = nullptr;
    double * normalization_term_positive_frequency = nullptr;
    #ifdef DEAC_TWO_SIDED_POPULATION
        double * normalization_term_negative_frequency = nullptr;
    #endif
    #ifdef USE_GPU
        double* d_normalization = nullptr;
        int* d_normalization_valid = nullptr;
        double* d_normalization_term_positive_frequency = nullptr;
        #ifdef DEAC_TWO_SIDED_POPULATION
            double* d_normalization_term_negative_frequency;
        #endif
    #endif
    if (normalize) {
        normalization = (double*) malloc(bytes_normalization);
        normalization_valid = (int*) malloc(bytes_normalization_valid);
        normalization_term_positive_frequency = (double*) malloc(bytes_normalization_term);
        #ifdef DEAC_TWO_SIDED_POPULATION
            normalization_term_negative_frequency = (double*) malloc(bytes_normalization_term);
        #endif
        const deac_numerics::NormalizationTerms normalization_terms =
                deac_numerics::make_normalization_terms(
                        std::span<const double>(frequency, genome_size),
                        frequency_weights, temperature);
        std::copy(
                normalization_terms.positive_frequency.begin(),
                normalization_terms.positive_frequency.end(),
                normalization_term_positive_frequency);
        #ifdef DEAC_TWO_SIDED_POPULATION
            std::copy(
                    normalization_terms.negative_frequency.begin(),
                    normalization_terms.negative_frequency.end(),
                    normalization_term_negative_frequency);
        #endif

        #ifdef USE_GPU
            //Load normalization terms onto GPU
            GPU_ASSERT(deac_malloc_device(double, d_normalization,                         population_size, default_stream));
            GPU_ASSERT(deac_malloc_device(int,    d_normalization_valid,                   population_size, default_stream));
            GPU_ASSERT(deac_malloc_device(double, d_normalization_term_positive_frequency, genome_size,     default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_normalization_term_positive_frequency, normalization_term_positive_frequency, bytes_normalization_term, default_stream));
            #ifdef DEAC_TWO_SIDED_POPULATION
                GPU_ASSERT(deac_malloc_device(double, d_normalization_term_negative_frequency, genome_size,     default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                GPU_ASSERT(deac_memcpy_host_to_device(d_normalization_term_negative_frequency, normalization_term_negative_frequency, bytes_normalization_term, default_stream));
            #endif
            GPU_ASSERT(deac_wait(default_stream));
        #endif

        //Set normalization
        #ifdef USE_GPU
            GPU_ASSERT(deac_memset(d_normalization, 0, bytes_normalization, default_stream));
            GPU_ASSERT(deac_wait(default_stream));

            #ifdef USE_BLAS
                gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0/zeroth_moment, d_population_old_positive_frequency, d_normalization_term_positive_frequency, 0.0, d_normalization);
                GPU_ASSERT(deac_wait(default_stream));
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0/zeroth_moment, d_population_old_negative_frequency, d_normalization_term_negative_frequency, 1.0, d_normalization);
                    GPU_ASSERT(deac_wait(default_stream));
                #endif
            #else
                gpu_deac_gemv(default_stream, population_size, genome_size, 1.0/zeroth_moment, d_population_old_positive_frequency, d_normalization_term_positive_frequency, 0.0, d_normalization);
                GPU_ASSERT(deac_wait(default_stream));
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_deac_gemv(default_stream, population_size, genome_size, 1.0/zeroth_moment, d_population_old_negative_frequency, d_normalization_term_negative_frequency, 1.0, d_normalization);
                    GPU_ASSERT(deac_wait(default_stream));
                #endif
            #endif

            GPU_ASSERT(deac_memset(
                    d_normalization_valid, 1,
                    bytes_normalization_valid, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            gpu_deac_dgmmDiv1D(stream_array[0],
                    d_population_old_positive_frequency,
                    d_normalization, population_size, genome_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                gpu_deac_dgmmDiv1D(stream_array[1 % MAX_GPU_STREAMS],
                        d_population_old_negative_frequency,
                        d_normalization, population_size, genome_size);
                #if MAX_GPU_STREAMS > 1
                    GPU_ASSERT(deac_wait(stream_array[1]));
                #endif
            #endif
            GPU_ASSERT(deac_wait(stream_array[0]));
            gpu_validate_normalization_rows(stream_array[0],
                    d_population_old_positive_frequency,
                    d_normalization, d_normalization_valid, zeroth_moment,
                    population_size, genome_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                gpu_validate_normalization_rows(stream_array[1 % MAX_GPU_STREAMS],
                        d_population_old_negative_frequency,
                        d_normalization, d_normalization_valid, zeroth_moment,
                        population_size, genome_size);
                #if MAX_GPU_STREAMS > 1
                    GPU_ASSERT(deac_wait(stream_array[1]));
                #endif
            #endif
            GPU_ASSERT(deac_wait(stream_array[0]));
            gpu_cleanup_invalid_normalization_rows(stream_array[0],
                    d_population_old_positive_frequency,
                    d_normalization_valid, population_size, genome_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                gpu_cleanup_invalid_normalization_rows(stream_array[0],
                        d_population_old_negative_frequency,
                        d_normalization_valid, population_size, genome_size);
            #endif
            GPU_ASSERT(deac_wait(stream_array[0]));
            GPU_ASSERT(deac_memcpy_device_to_host(
                    normalization_valid, d_normalization_valid,
                    bytes_normalization_valid, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            if (!std::all_of(
                    normalization_valid, normalization_valid + population_size,
                    [](int valid) { return valid != 0; })) {
                fail_with_error(
                        "initial population contains an unrepresentable "
                        "normalization row");
            }
        #else
            for (size_t i=0; i<population_size; i++) {
                normalization[i] = 0.0;
            }
            matrix_multiply_MxN_by_Nx1(normalization, population_old_positive_frequency,
                    normalization_term_positive_frequency, population_size, genome_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                matrix_multiply_MxN_by_Nx1(normalization, population_old_negative_frequency,
                        normalization_term_negative_frequency, population_size, genome_size);
            #endif
            for (size_t i=0; i<population_size; i++) {
                const bool valid = deac_numerics::try_apply_normalization(
                        zeroth_moment, normalization[i],
                        std::span<double>(
                                population_old_positive_frequency + i*genome_size,
                                genome_size)
                        #ifdef DEAC_TWO_SIDED_POPULATION
                            , std::span<double>(
                                    population_old_negative_frequency + i*genome_size,
                                    genome_size)
                        #endif
                        );
                if (!valid) {
                    fail_with_error(
                            "initial population contains an unrepresentable "
                            "normalization row");
                }
            }
        #endif
    }

    // Set first moment term
    size_t bytes_first_moments_term = sizeof(double)*genome_size;
    size_t bytes_first_moments = sizeof(double)*population_size;

    double * first_moments = nullptr;
    double * first_moments_term_positive_frequency = nullptr;
    #ifdef DEAC_TWO_SIDED_POPULATION
        double * first_moments_term_negative_frequency = nullptr;
    #endif
    #ifdef USE_GPU
        double* d_first_moments;
        double* d_first_moments_term_positive_frequency;
        #ifdef DEAC_TWO_SIDED_POPULATION
            double* d_first_moments_term_negative_frequency;
        #endif
    #endif

    if (use_first_moment) {
        first_moments_term_positive_frequency = (double*) malloc(bytes_first_moments_term);
        #ifdef DEAC_TWO_SIDED_POPULATION
            first_moments_term_negative_frequency = (double*) malloc(bytes_first_moments_term);
        #endif
        for (size_t j=0; j<genome_size; j++) {
            double f = frequency[j];
            const double df = frequency_weights[j];
            #ifndef ZEROT
                #ifdef USE_HYPERBOLIC_MODEL
                    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        first_moments_term_positive_frequency[j] = df*f*sinh(0.5*beta*f);
                    #else
                        first_moments_term_positive_frequency[j] = 0.5*df*f/exp(-0.5*beta*f);
                        first_moments_term_negative_frequency[j] = -0.5*df*f*exp(-0.5*beta*f);
                    #endif
                #endif
                #ifdef USE_STANDARD_MODEL
                    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        first_moments_term_positive_frequency[j] = df*f*(1.0 - exp(-beta*f));
                    #else
                        first_moments_term_positive_frequency[j] = df*f;
                        first_moments_term_negative_frequency[j] = -df*f;
                    #endif
                #endif
                #ifdef USE_NORMALIZATION_MODEL
                    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        first_moments_term_positive_frequency[j] = df*f*tanh(0.5*beta*f);
                    #else
                        first_moments_term_positive_frequency[j] = df*f*(1.0/(1.0 + exp(-beta*f)));
                        double e_to_bf = exp(-beta*f); // exp(beta*f) for negative f
                        first_moments_term_negative_frequency[j] = -df*f*(e_to_bf/(1.0 + e_to_bf));
                    #endif
                #endif
            #else
                first_moments_term_positive_frequency[j] = df*f;
                #ifdef DEAC_TWO_SIDED_POPULATION
                    first_moments_term_negative_frequency[j] = -df*f;
                #endif
            #endif
        }

        first_moments = (double*) malloc(bytes_first_moments);
        #ifdef USE_GPU
            GPU_ASSERT(deac_malloc_device(double, d_first_moments,                         population_size, default_stream));
            GPU_ASSERT(deac_malloc_device(double, d_first_moments_term_positive_frequency, genome_size,     default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_first_moments_term_positive_frequency, first_moments_term_positive_frequency, bytes_first_moments_term, default_stream));
            #ifdef DEAC_TWO_SIDED_POPULATION
                GPU_ASSERT(deac_malloc_device(double, d_first_moments_term_negative_frequency, genome_size,     default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                GPU_ASSERT(deac_memcpy_host_to_device(d_first_moments_term_negative_frequency, first_moments_term_negative_frequency, bytes_first_moments_term, default_stream));
            #endif
            GPU_ASSERT(deac_wait(default_stream));

            GPU_ASSERT(deac_memset(d_first_moments, 0, bytes_first_moments, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            #ifdef USE_BLAS
                gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0, d_population_old_positive_frequency, d_first_moments_term_positive_frequency, 0.0, d_first_moments);
                GPU_ASSERT(deac_wait(default_stream));
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0, d_population_old_negative_frequency, d_first_moments_term_negative_frequency, 1.0, d_first_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                #endif
            #else
                gpu_deac_gemv(default_stream, population_size, genome_size, 1.0, d_population_old_positive_frequency, d_first_moments_term_positive_frequency, 0.0, d_first_moments);
                GPU_ASSERT(deac_wait(default_stream));
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_deac_gemv(default_stream, population_size, genome_size, 1.0, d_population_old_negative_frequency, d_first_moments_term_negative_frequency, 1.0, d_first_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                #endif
            #endif
        #else
            for (size_t i=0; i<population_size; i++) {
                first_moments[i] = 0.0;
            }
            matrix_multiply_MxN_by_Nx1(first_moments, population_old_positive_frequency,
                    first_moments_term_positive_frequency, population_size, genome_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                matrix_multiply_MxN_by_Nx1(first_moments, population_old_negative_frequency,
                        first_moments_term_negative_frequency, population_size, genome_size);
            #endif
        #endif
    }

    // Set third moment term
    size_t bytes_third_moments_term = sizeof(double)*genome_size;
    size_t bytes_third_moments = sizeof(double)*population_size;

    double * third_moments = nullptr;
    double * third_moments_term_positive_frequency = nullptr;
    #ifdef DEAC_TWO_SIDED_POPULATION
        double * third_moments_term_negative_frequency = nullptr;
    #endif
    #ifdef USE_GPU
        double* d_third_moments;
        double* d_third_moments_term_positive_frequency;
        #ifdef DEAC_TWO_SIDED_POPULATION
            double* d_third_moments_term_negative_frequency;
        #endif
    #endif

    if (use_third_moment) {
        third_moments_term_positive_frequency = (double*) malloc(bytes_third_moments_term);
        #ifdef DEAC_TWO_SIDED_POPULATION
            third_moments_term_negative_frequency = (double*) malloc(bytes_third_moments_term);
        #endif
        for (size_t j=0; j<genome_size; j++) {
            double f = frequency[j];
            const double df = frequency_weights[j];
            #ifndef ZEROT
                #ifdef USE_HYPERBOLIC_MODEL
                    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        third_moments_term_positive_frequency[j] = df*pow(f,3)*f*sinh(0.5*beta*f);
                    #else
                        third_moments_term_positive_frequency[j] = 0.5*df*pow(f,3)*f/exp(-0.5*beta*f);
                        third_moments_term_negative_frequency[j] = -0.5*df*pow(f,3)*f*exp(-0.5*beta*f);
                    #endif
                #endif
                #ifdef USE_STANDARD_MODEL
                    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        third_moments_term_positive_frequency[j] = df*pow(f,3)*f*(1.0 - exp(-beta*f));
                    #else
                        third_moments_term_positive_frequency[j] = df*pow(f,3)*f;
                        third_moments_term_negative_frequency[j] = -df*pow(f,3)*f;
                    #endif
                #endif
                #ifdef USE_NORMALIZATION_MODEL
                    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        third_moments_term_positive_frequency[j] = df*pow(f,3)*f*tanh(0.5*beta*f);
                    #else
                        third_moments_term_positive_frequency[j] = df*pow(f,3)*f*(1.0/(1.0 + exp(-beta*f)));
                        double e_to_bf = exp(-beta*f); // exp(beta*f) for negative f
                        third_moments_term_negative_frequency[j] = -df*pow(f,3)*f*(e_to_bf/(1.0 + e_to_bf));
                    #endif
                #endif
            #else
                third_moments_term_positive_frequency[j] =
                        deac_numerics::zero_temperature_third_moment_term(f, df);
            #endif
        }

        third_moments = (double*) malloc(bytes_third_moments);
        #ifdef USE_GPU
            GPU_ASSERT(deac_malloc_device(double, d_third_moments,                         population_size, default_stream));
            GPU_ASSERT(deac_malloc_device(double, d_third_moments_term_positive_frequency, genome_size,     default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_third_moments_term_positive_frequency, third_moments_term_positive_frequency, bytes_third_moments_term, default_stream));
            #ifdef DEAC_TWO_SIDED_POPULATION
                GPU_ASSERT(deac_malloc_device(double, d_third_moments_term_negative_frequency, genome_size,     default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                GPU_ASSERT(deac_memcpy_host_to_device(d_third_moments_term_negative_frequency, third_moments_term_negative_frequency, bytes_third_moments_term, default_stream));
            #endif
            GPU_ASSERT(deac_wait(default_stream));

            GPU_ASSERT(deac_memset(d_third_moments, 0, bytes_third_moments, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            #ifdef USE_BLAS
                gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0, d_population_old_positive_frequency, d_third_moments_term_positive_frequency, 0.0, d_third_moments);
                GPU_ASSERT(deac_wait(default_stream));
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0, d_population_old_negative_frequency, d_third_moments_term_negative_frequency, 1.0, d_third_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                #endif
            #else
                gpu_deac_gemv(default_stream, population_size, genome_size, 1.0, d_population_old_positive_frequency, d_third_moments_term_positive_frequency, 0.0, d_third_moments);
                GPU_ASSERT(deac_wait(default_stream));
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_deac_gemv(default_stream, population_size, genome_size, 1.0, d_population_old_negative_frequency, d_third_moments_term_negative_frequency, 1.0, d_third_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                #endif
            #endif
        #else
            for (size_t i=0; i<population_size; i++) {
                third_moments[i] = 0.0;
            }
            matrix_multiply_MxN_by_Nx1(third_moments, population_old_positive_frequency,
                    third_moments_term_positive_frequency, population_size, genome_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                matrix_multiply_MxN_by_Nx1(third_moments, population_old_negative_frequency,
                        third_moments_term_negative_frequency, population_size, genome_size);
            #endif
        #endif
    }

    //Set isf_model and calculate fitness
    double * isf_model;
    #ifdef USE_GPU
        double* d_isf_model;
    #endif
    size_t bytes_isf_model = sizeof(double)*number_of_timeslices*population_size;
    isf_model = (double*) malloc(bytes_isf_model);
    #ifdef USE_GPU
        GPU_ASSERT(deac_malloc_device(double, d_isf_model, number_of_timeslices*population_size, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        GPU_ASSERT(deac_memset(d_isf_model, 0, bytes_isf_model, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        #ifdef USE_BLAS
            gpu_blas_gemm(default_blas_handle, population_size, number_of_timeslices, genome_size, 1.0, d_population_old_positive_frequency, d_isf_term_positive_frequency, 0.0, d_isf_model);
            GPU_ASSERT(deac_wait(default_stream));
            #ifdef DEAC_TWO_SIDED_POPULATION
                gpu_blas_gemm(default_blas_handle, population_size, number_of_timeslices, genome_size, 1.0, d_population_old_negative_frequency, d_isf_term_negative_frequency, 1.0, d_isf_model);
                GPU_ASSERT(deac_wait(default_stream));
            #endif
        #else
            gpu_matmul(default_stream, population_size, number_of_timeslices, genome_size, 1.0, d_population_old_positive_frequency, d_isf_term_positive_frequency, 0.0, d_isf_model);
            GPU_ASSERT(deac_wait(default_stream));
            #ifdef DEAC_TWO_SIDED_POPULATION
                gpu_matmul(default_stream, population_size, number_of_timeslices, genome_size, 1.0, d_population_old_negative_frequency, d_isf_term_negative_frequency, 1.0, d_isf_model);
                GPU_ASSERT(deac_wait(default_stream));
            #endif
        #endif
    #else
        for (size_t i=0; i<population_size*number_of_timeslices; i++) {
            isf_model[i] = 0.0;
        }
        deac_numerics::accumulate_population_projection(
                isf_model, isf_term_positive_frequency, population_old_positive_frequency,
                number_of_timeslices, genome_size, population_size);
        #ifdef DEAC_TWO_SIDED_POPULATION
            deac_numerics::accumulate_population_projection(
                    isf_model, isf_term_negative_frequency, population_old_negative_frequency,
                    number_of_timeslices, genome_size, population_size);
        #endif
    #endif

    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        double * negative_first_moments_term;
        double * negative_first_moments;
        #ifdef USE_GPU
            double* d_negative_first_moments_term;
            double* d_negative_first_moments;
        #endif
        size_t bytes_negative_first_moments_term = sizeof(double)*number_of_timeslices;
        size_t bytes_negative_first_moments = sizeof(double)*population_size;
        double negative_first_moment = 0.0;
        double negative_first_moment_error = 0.0;
        if (use_negative_first_moment) {
            negative_first_moments_term = (double*) malloc(bytes_negative_first_moments_term);
            const std::vector<double> imaginary_time_weights = deac_numerics::trapezoidal_weights(
                    std::span<const double>(imaginary_time, number_of_timeslices));
            for (size_t j=0; j<number_of_timeslices; j++) {
                const double dt = imaginary_time_weights[j];
                negative_first_moments_term[j] = dt;
                negative_first_moment += isf[j]*dt;
                negative_first_moment_error += pow(isf_error[j],2) * pow(dt,2);
            }
            negative_first_moment_error = sqrt(negative_first_moment_error);

            negative_first_moments = (double*) malloc(bytes_negative_first_moments);
            #ifdef USE_GPU
                GPU_ASSERT(deac_malloc_device(double, d_negative_first_moments_term, number_of_timeslices, default_stream));
                GPU_ASSERT(deac_malloc_device(double, d_negative_first_moments,      population_size,      default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                GPU_ASSERT(deac_memcpy_host_to_device(d_negative_first_moments_term, negative_first_moments_term, bytes_negative_first_moments_term, default_stream));
                GPU_ASSERT(deac_wait(default_stream));

                GPU_ASSERT(deac_memset(d_negative_first_moments, 0, bytes_negative_first_moments, default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                #ifdef USE_BLAS
                    gpu_blas_gemv(default_blas_handle, population_size, number_of_timeslices, 1.0, d_isf_model, d_negative_first_moments_term, 0.0, d_negative_first_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                #else
                    gpu_deac_gemv(default_stream, population_size, number_of_timeslices, 1.0, d_isf_model, d_negative_first_moments_term, 0.0, d_negative_first_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                #endif
            #else
                for (size_t i=0; i<population_size; i++) {
                    negative_first_moments[i] = 0.0;
                }
                matrix_multiply_MxN_by_Nx1(negative_first_moments, isf_model,
                        negative_first_moments_term, population_size, number_of_timeslices);
            #endif
        }
    #else
        //FIXME inverse moment not implemented, need lim freq --> 0 or detailed balance condition or to do math
    #endif

    double * fitness_old;
    size_t bytes_fitness = sizeof(double)*population_size;
    fitness_old = (double*) malloc(bytes_fitness);

    #ifdef USE_GPU
        double* d_fitness_old;
        double* d_fitness_new;
        GPU_ASSERT(deac_malloc_device(double, d_fitness_old, population_size, default_stream));
        GPU_ASSERT(deac_malloc_device(double, d_fitness_new, population_size, default_stream));
        GPU_ASSERT(deac_wait(default_stream));

        #ifdef DEAC_TEST_POISON_GPU_FITNESS
            // Test-only full-solver seam: poison both destinations at their
            // production allocation point, before either scoring pass can
            // define them.  The shared GPU status path evaluates and checks
            // each call exactly once in every build type.
            std::fill(
                    fitness_old, fitness_old + population_size,
                    std::numeric_limits<double>::quiet_NaN());
            DEAC_TEST_GPU_CALL(deac_memcpy_host_to_device(
                    d_fitness_old, fitness_old,
                    bytes_fitness, default_stream));
            DEAC_TEST_GPU_CALL(deac_memcpy_host_to_device(
                    d_fitness_new, fitness_old,
                    bytes_fitness, default_stream));
            DEAC_TEST_GPU_CALL(deac_wait(default_stream));
            const auto test_require_poisoned_gpu_fitness =
                    [&](double* device_fitness, const char* buffer) {
                std::fill(
                        fitness_old, fitness_old + population_size, 0.0);
                DEAC_TEST_GPU_CALL(deac_memcpy_device_to_host(
                        fitness_old, device_fitness,
                        bytes_fitness, default_stream));
                DEAC_TEST_GPU_CALL(deac_wait(default_stream));
                if (!std::all_of(
                        fitness_old, fitness_old + population_size,
                        [](double value) { return std::isnan(value); })) {
                    fail_with_error(
                            std::string("GPU fitness buffer was not poisoned: ")
                            + buffer);
                }
            };
            test_require_poisoned_gpu_fitness(d_fitness_old, "old");
            test_require_poisoned_gpu_fitness(d_fitness_new, "new");
            std::cout << "test_poisoned_gpu_fitness_buffers: old,new\n";
            const auto test_require_finite_gpu_fitness =
                    [&](double* device_fitness, const char* phase) {
                std::fill(
                        fitness_old, fitness_old + population_size,
                        std::numeric_limits<double>::quiet_NaN());
                DEAC_TEST_GPU_CALL(deac_memcpy_device_to_host(
                        fitness_old, device_fitness,
                        bytes_fitness, default_stream));
                DEAC_TEST_GPU_CALL(deac_wait(default_stream));
                if (!std::all_of(
                        fitness_old, fitness_old + population_size,
                        [](double value) { return std::isfinite(value); })) {
                    fail_with_error(
                            std::string("poisoned GPU fitness remained non-finite ")
                            + phase);
                }
                std::cout << "test_poisoned_gpu_fitness_" << phase
                          << ": finite\n";
            };
        #endif

        gpu_deac_reduced_chi_squared(default_stream, d_isf_model, d_isf, d_isf_error, d_fitness_old, population_size, number_of_timeslices, 0, 0.0);
        GPU_ASSERT(deac_wait(default_stream));

        #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            if (use_negative_first_moment) {
                gpu_deac_add_scalar_reduced_chi_squared(default_stream, d_negative_first_moments, negative_first_moment, negative_first_moment_error, d_fitness_old, population_size);
                GPU_ASSERT(deac_wait(default_stream));
            }
        #else
            //FIXME inverse first moment not implemented
        #endif
        if (use_first_moment) {
            gpu_deac_add_scalar_reduced_chi_squared(default_stream, d_first_moments, first_moment, first_moment_error, d_fitness_old, population_size);
            GPU_ASSERT(deac_wait(default_stream));
        }
        if (use_third_moment) {
            gpu_deac_add_scalar_reduced_chi_squared(default_stream, d_third_moments, third_moment, third_moment_error, d_fitness_old, population_size);
            GPU_ASSERT(deac_wait(default_stream));
        }
        #ifdef DEAC_TEST_POISON_GPU_FITNESS
            test_require_finite_gpu_fitness(d_fitness_old, "initial");
        #endif
    #else
        for (size_t i=0; i<population_size; i++) {
            double _fitness = reduced_chi_square_statistic(isf,
                    isf_model + i*number_of_timeslices, isf_error,
                    number_of_timeslices)/number_of_timeslices;
            #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                if (use_negative_first_moment) {
                    _fitness += pow((negative_first_moment - negative_first_moments[i])/negative_first_moment_error,2);
                }
            #else
                //FIXME inverse first moment not implemented
            #endif
            if (use_first_moment) {
                _fitness += deac_numerics::scalar_chi_square_penalty(
                        first_moments[i], first_moment, first_moment_error);
            }
            if (use_third_moment) {
                _fitness += pow((third_moment - third_moments[i])/third_moment_error,2);
            }
            fitness_old[i] = _fitness;
        }
    #endif

    size_t bytes_crossover_probabilities = sizeof(double)*population_size;
    double * crossover_probabilities_old_positive_frequency;
    double * crossover_probabilities_new_positive_frequency;
    crossover_probabilities_old_positive_frequency = (double*) malloc(bytes_crossover_probabilities);
    crossover_probabilities_new_positive_frequency = (double*) malloc(bytes_crossover_probabilities);
    #ifdef DEAC_TWO_SIDED_POPULATION
        double * crossover_probabilities_old_negative_frequency;
        double * crossover_probabilities_new_negative_frequency;
        crossover_probabilities_old_negative_frequency = (double*) malloc(bytes_crossover_probabilities);
        crossover_probabilities_new_negative_frequency = (double*) malloc(bytes_crossover_probabilities);
    #endif

    for (size_t i=0; i<population_size; i++) {
        crossover_probabilities_old_positive_frequency[i] = crossover_probability;
        #ifdef DEAC_TWO_SIDED_POPULATION
            crossover_probabilities_old_negative_frequency[i] = crossover_probability;
        #endif
    }

    #ifdef USE_GPU
        double* d_crossover_probabilities_old_positive_frequency;
        double* d_crossover_probabilities_new_positive_frequency;
        GPU_ASSERT(deac_malloc_device(double, d_crossover_probabilities_old_positive_frequency, population_size, default_stream));
        GPU_ASSERT(deac_malloc_device(double, d_crossover_probabilities_new_positive_frequency, population_size, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_crossover_probabilities_old_positive_frequency, crossover_probabilities_old_positive_frequency, bytes_crossover_probabilities, default_stream));
        #ifdef DEAC_TWO_SIDED_POPULATION
            double* d_crossover_probabilities_old_negative_frequency;
            double* d_crossover_probabilities_new_negative_frequency;
            GPU_ASSERT(deac_malloc_device(double, d_crossover_probabilities_old_negative_frequency, population_size, default_stream));
            GPU_ASSERT(deac_malloc_device(double, d_crossover_probabilities_new_negative_frequency, population_size, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_crossover_probabilities_old_negative_frequency, crossover_probabilities_old_negative_frequency, bytes_crossover_probabilities, default_stream));
        #endif
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    size_t bytes_differential_weights = sizeof(double)*population_size;
    double * differential_weights_old_positive_frequency;
    double * differential_weights_new_positive_frequency;
    differential_weights_old_positive_frequency = (double*) malloc(bytes_differential_weights);
    differential_weights_new_positive_frequency = (double*) malloc(bytes_differential_weights);
    #ifdef DEAC_TWO_SIDED_POPULATION
        double * differential_weights_old_negative_frequency;
        double * differential_weights_new_negative_frequency;
        differential_weights_old_negative_frequency = (double*) malloc(bytes_differential_weights);
        differential_weights_new_negative_frequency = (double*) malloc(bytes_differential_weights);
    #endif

    for (size_t i=0; i<population_size; i++) {
        differential_weights_old_positive_frequency[i] = differential_weight;
        #ifdef DEAC_TWO_SIDED_POPULATION
            differential_weights_old_negative_frequency[i] = differential_weight;
        #endif
    }

    #ifdef USE_GPU
        double* d_differential_weights_old_positive_frequency;
        double* d_differential_weights_new_positive_frequency;
        GPU_ASSERT(deac_malloc_device(double, d_differential_weights_old_positive_frequency, population_size, default_stream));
        GPU_ASSERT(deac_malloc_device(double, d_differential_weights_new_positive_frequency, population_size, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_differential_weights_old_positive_frequency, differential_weights_old_positive_frequency, bytes_differential_weights, default_stream));
        #ifdef DEAC_TWO_SIDED_POPULATION
            double* d_differential_weights_old_negative_frequency;
            double* d_differential_weights_new_negative_frequency;
            GPU_ASSERT(deac_malloc_device(double, d_differential_weights_old_negative_frequency, population_size, default_stream));
            GPU_ASSERT(deac_malloc_device(double, d_differential_weights_new_negative_frequency, population_size, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_differential_weights_old_negative_frequency, differential_weights_old_negative_frequency, bytes_differential_weights, default_stream));
        #endif
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    //Initialize statistics arrays
    double* fitness_mean = nullptr;
    double* fitness_minimum = nullptr;
    double* fitness_squared_mean = nullptr;
    #ifdef USE_GPU
        double* d_fitness_mean;
        double* d_fitness_squared_mean;
    #endif
    size_t bytes_fitness_mean = sizeof(double)*number_of_generations;
    size_t bytes_fitness_minimum = sizeof(double)*number_of_generations;
    size_t bytes_fitness_squared_mean = sizeof(double)*number_of_generations;
    if (track_stats) {
        fitness_mean = (double*) malloc(bytes_fitness_mean);
        fitness_minimum = (double*) malloc(bytes_fitness_minimum);
        fitness_squared_mean = (double*) malloc(bytes_fitness_squared_mean);
        #ifdef USE_GPU
            GPU_ASSERT(deac_malloc_device(double, d_fitness_mean        , number_of_generations, default_stream));
            GPU_ASSERT(deac_malloc_device(double, d_fitness_squared_mean, number_of_generations, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            GPU_ASSERT(deac_memset(d_fitness_mean,         0, bytes_fitness_mean,         default_stream));
            GPU_ASSERT(deac_memset(d_fitness_squared_mean, 0, bytes_fitness_squared_mean, default_stream));
        #endif
    }
    
    size_t bytes_mutate_indices = sizeof(bool)*genome_size*population_size;
    bool* mutate_indices_positive_frequency;
    mutate_indices_positive_frequency = (bool*) malloc(bytes_mutate_indices);
    #ifdef DEAC_TWO_SIDED_POPULATION
        bool* mutate_indices_negative_frequency;
        mutate_indices_negative_frequency = (bool*) malloc(bytes_mutate_indices);
    #endif
    #ifdef USE_GPU
        bool* d_mutate_indices_positive_frequency;
        GPU_ASSERT(deac_malloc_device(bool, d_mutate_indices_positive_frequency, population_size*genome_size, default_stream));
        #ifdef DEAC_TWO_SIDED_POPULATION
            bool* d_mutate_indices_negative_frequency;
            GPU_ASSERT(deac_malloc_device(bool, d_mutate_indices_negative_frequency, population_size*genome_size, default_stream));
        #endif

        bool* d_rejection_indices;
        GPU_ASSERT(deac_malloc_device(bool, d_rejection_indices, population_size, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    size_t* mutant_indices;
    size_t bytes_mutant_indices = sizeof(size_t)*3*population_size;
    mutant_indices = (size_t*) malloc(bytes_mutant_indices);
    #ifdef USE_GPU
        size_t* d_mutant_indices;
        GPU_ASSERT(deac_malloc_device(size_t, d_mutant_indices, 3*population_size, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    double minimum_fitness;
    size_t minimum_fitness_idx;
    #ifdef USE_GPU
        size_t bytes_minimum_fitness = sizeof(double);
        double* d_minimum_fitness;
        GPU_ASSERT(deac_malloc_device(double, d_minimum_fitness, 1, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    #ifdef USE_GPU
        #ifdef DEAC_TWO_SIDED_POPULATION
            size_t size_rng_state = 8*population_size*(genome_size + 1);
        #else
            size_t size_rng_state  = 4*population_size*(genome_size + 1);
        #endif
        size_t bytes_rng_state = sizeof(uint64_t)*size_rng_state;

        // Generate rng state
        uint64_t* d_rng_state;
        uint64_t* rng_state;
        rng_state = (uint64_t *) malloc(bytes_rng_state);

        for (size_t i=0; i < size_rng_state/4; i++) {
            xoshiro256p_copy_state(rng_state + 4*i, rng->s);
            xoshiro256p_jump(rng->s);
        }
        GPU_ASSERT(deac_malloc_device(uint64_t, d_rng_state, size_rng_state, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_rng_state, rng_state, bytes_rng_state, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
    #endif
    
    size_t generation = 0;
    for (size_t ii=0; ii < number_of_generations - 1; ii++) {
        generation = ii;
        #ifdef USE_GPU
            gpu_get_minimum(default_stream, d_minimum_fitness, d_fitness_old, population_size);
            GPU_ASSERT(deac_wait(default_stream));
            GPU_ASSERT(deac_memcpy_device_to_host(&minimum_fitness, d_minimum_fitness, bytes_minimum_fitness, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
        #else
            minimum_fitness = minimum(fitness_old, population_size);
        #endif

        //Get Statistics
        if (track_stats) {
            #ifdef USE_GPU
                gpu_set_fitness_mean(default_stream, d_fitness_mean + ii, d_fitness_old, population_size);
                fitness_minimum[ii] = minimum_fitness;
                gpu_set_fitness_squared_mean(default_stream, d_fitness_squared_mean + ii, d_fitness_old, population_size);
                GPU_ASSERT(deac_wait(default_stream));
            #else
                fitness_mean[ii] = mean(fitness_old, population_size);
                fitness_minimum[ii] = minimum_fitness;
                fitness_squared_mean[ii] = squared_mean(fitness_old, population_size);
            #endif
        }
        
        //Stopping criteria
        if (minimum_fitness <= stop_minimum_fitness) {
            break;
        }

        #ifdef USE_GPU
            size_t grid_size_self_adapting_parameters = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            gpu_set_crossover_probabilities_new(stream_array[0], grid_size_self_adapting_parameters, d_rng_state, d_crossover_probabilities_new_positive_frequency, d_crossover_probabilities_old_positive_frequency, self_adapting_crossover_probability, population_size);
            gpu_set_differential_weights_new(stream_array[1 % MAX_GPU_STREAMS], grid_size_self_adapting_parameters, d_rng_state + 4*population_size, d_differential_weights_new_positive_frequency, d_differential_weights_old_positive_frequency, self_adapting_differential_weight_probability, population_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                gpu_set_crossover_probabilities_new(stream_array[2 % MAX_GPU_STREAMS], grid_size_self_adapting_parameters, d_rng_state + 8*population_size, d_crossover_probabilities_new_negative_frequency, d_crossover_probabilities_old_negative_frequency, self_adapting_crossover_probability, population_size);
                gpu_set_differential_weights_new(stream_array[3 % MAX_GPU_STREAMS], grid_size_self_adapting_parameters, d_rng_state + 12*population_size, d_differential_weights_new_negative_frequency, d_differential_weights_old_negative_frequency, self_adapting_differential_weight_probability, population_size);
            #endif
            for (auto& s : stream_array) {
                GPU_ASSERT(deac_wait(s));
            }
        #else
            //Set crossover probabilities and differential weights
            for (size_t i=0; i<population_size; i++) {
                if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
                    crossover_probabilities_new_positive_frequency[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53;
                } else {
                    crossover_probabilities_new_positive_frequency[i] = crossover_probabilities_old_positive_frequency[i];
                }

                if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
                    differential_weights_new_positive_frequency[i] = 2.0*((xoshiro256p(rng) >> 11) * 0x1.0p-53);
                } else {
                    differential_weights_new_positive_frequency[i] = differential_weights_old_positive_frequency[i];
                }

                #ifdef DEAC_TWO_SIDED_POPULATION
                    if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
                        crossover_probabilities_new_negative_frequency[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53;
                    } else {
                        crossover_probabilities_new_negative_frequency[i] = crossover_probabilities_old_negative_frequency[i];
                    }

                    if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
                        differential_weights_new_negative_frequency[i] = 2.0*((xoshiro256p(rng) >> 11) * 0x1.0p-53);
                    } else {
                        differential_weights_new_negative_frequency[i] = differential_weights_old_negative_frequency[i];
                    }
                #endif
            }
        #endif

        #ifdef USE_GPU
            size_t grid_size_set_mutant_indices = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            size_t grid_size_set_mutate_indices = (population_size*genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            gpu_set_mutant_indices(stream_array[0], grid_size_set_mutant_indices, d_rng_state, d_mutant_indices, population_size);
            gpu_set_mutate_indices(stream_array[1 % MAX_GPU_STREAMS], grid_size_set_mutate_indices, d_rng_state + 4*population_size, d_mutate_indices_positive_frequency, d_crossover_probabilities_new_positive_frequency, population_size, genome_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                gpu_set_mutate_indices(stream_array[2 % MAX_GPU_STREAMS], grid_size_set_mutate_indices, d_rng_state + 4*population_size + 4*population_size*genome_size, d_mutate_indices_negative_frequency, d_crossover_probabilities_new_negative_frequency, population_size, genome_size);
            #endif
            for (auto& s : stream_array) {
                GPU_ASSERT(deac_wait(s));
            }
        #else
            //Set mutant population and indices 
            for (size_t i=0; i<population_size; i++) {
                set_mutant_indices(rng, mutant_indices + 3*i, i, population_size);
                double crossover_rate_positive_frequency = crossover_probabilities_new_positive_frequency[i];
                const uint64_t crossover_threshold_positive_frequency =
                        probability_threshold(crossover_rate_positive_frequency);
                for (size_t j=0; j<genome_size; j++) {
                    mutate_indices_positive_frequency[i*genome_size + j] =
                            (xoshiro256p(rng) >> 11) < crossover_threshold_positive_frequency;
                }
                #ifdef DEAC_TWO_SIDED_POPULATION
                    double crossover_rate_negative_frequency = crossover_probabilities_new_negative_frequency[i];
                    const uint64_t crossover_threshold_negative_frequency =
                            probability_threshold(crossover_rate_negative_frequency);
                    for (size_t j=0; j<genome_size; j++) {
                        mutate_indices_negative_frequency[i*genome_size + j] =
                                (xoshiro256p(rng) >> 11) < crossover_threshold_negative_frequency;
                    }
                #endif
            }
        #endif

        #ifdef USE_GPU
            size_t grid_size_set_population_new = (population_size*genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            gpu_set_population_new(stream_array[0], grid_size_set_population_new, d_population_new_positive_frequency, d_population_old_positive_frequency, d_mutant_indices, d_differential_weights_new_positive_frequency, d_mutate_indices_positive_frequency, population_size, genome_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                gpu_set_population_new(stream_array[1 % MAX_GPU_STREAMS], grid_size_set_population_new, d_population_new_negative_frequency, d_population_old_negative_frequency, d_mutant_indices, d_differential_weights_new_negative_frequency, d_mutate_indices_negative_frequency, population_size, genome_size);
                #if MAX_GPU_STREAMS > 1
                    GPU_ASSERT(deac_wait(stream_array[1]));
                #endif
            #endif
            GPU_ASSERT(deac_wait(stream_array[0]));

            #ifdef DEAC_TWO_SIDED_POPULATION
                size_t grid_size_match_population_zero = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                gpu_match_population_zero(default_stream, grid_size_match_population_zero, d_population_new_negative_frequency, d_population_new_positive_frequency, population_size, genome_size);
                GPU_ASSERT(deac_wait(default_stream));
            #endif
        #else
            #ifdef ALLOW_NEGATIVE_SPECTRAL_WEIGHT
                constexpr bool allow_negative_spectral_weight = true;
            #else
                constexpr bool allow_negative_spectral_weight = false;
            #endif
            for (size_t i=0; i<population_size; i++) {
                double F_positive_frequency = differential_weights_new_positive_frequency[i];
                #ifdef DEAC_TWO_SIDED_POPULATION
                    double F_negative_frequency = differential_weights_new_negative_frequency[i];
                #endif
                size_t mutant_index1 = mutant_indices[3*i];
                size_t mutant_index2 = mutant_indices[3*i + 1];
                size_t mutant_index3 = mutant_indices[3*i + 2];
                deac_numerics::form_trial_population_row<
                        allow_negative_spectral_weight>(
                        population_new_positive_frequency + i*genome_size,
                        population_old_positive_frequency + i*genome_size,
                        population_old_positive_frequency + mutant_index1*genome_size,
                        population_old_positive_frequency + mutant_index2*genome_size,
                        population_old_positive_frequency + mutant_index3*genome_size,
                        mutate_indices_positive_frequency + i*genome_size,
                        F_positive_frequency,
                        genome_size);
                #ifdef DEAC_TWO_SIDED_POPULATION
                    deac_numerics::form_trial_population_row<
                            allow_negative_spectral_weight>(
                            population_new_negative_frequency + i*genome_size,
                            population_old_negative_frequency + i*genome_size,
                            population_old_negative_frequency + mutant_index1*genome_size,
                            population_old_negative_frequency + mutant_index2*genome_size,
                            population_old_negative_frequency + mutant_index3*genome_size,
                            mutate_indices_negative_frequency + i*genome_size,
                            F_negative_frequency,
                            genome_size);
                    deac_numerics::couple_trial_population_zero(
                            population_new_negative_frequency + i*genome_size,
                            population_new_positive_frequency + i*genome_size);
                #endif
            }
        #endif

        // Normalization
        if (normalize) {
            #ifdef USE_GPU
                #ifdef USE_BLAS
                    gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0/zeroth_moment, d_population_new_positive_frequency, d_normalization_term_positive_frequency, 0.0, d_normalization);
                    GPU_ASSERT(deac_wait(default_stream));
                    #ifdef DEAC_TWO_SIDED_POPULATION
                        gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0/zeroth_moment, d_population_new_negative_frequency, d_normalization_term_negative_frequency, 1.0, d_normalization);
                        GPU_ASSERT(deac_wait(default_stream));
                    #endif
                #else
                    gpu_deac_gemv(default_stream, population_size, genome_size, 1.0/zeroth_moment, d_population_new_positive_frequency, d_normalization_term_positive_frequency, 0.0, d_normalization);
                    GPU_ASSERT(deac_wait(default_stream));
                    #ifdef DEAC_TWO_SIDED_POPULATION
                        gpu_deac_gemv(default_stream, population_size, genome_size, 1.0/zeroth_moment, d_population_new_negative_frequency, d_normalization_term_negative_frequency, 1.0, d_normalization);
                        GPU_ASSERT(deac_wait(default_stream));
                    #endif
                #endif

                #ifdef DEAC_TEST_FORCE_INVALID_NORMALIZATION_TRIALS
                    // Test-only executable: deterministically exercise the
                    // complete rejection path for degenerate evolved rows.
                    GPU_ASSERT(deac_memset(
                            d_normalization, 0,
                            bytes_normalization, default_stream));
                    GPU_ASSERT(deac_wait(default_stream));
                #endif

                GPU_ASSERT(deac_memset(
                        d_normalization_valid, 1,
                        bytes_normalization_valid, default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                gpu_deac_dgmmDiv1D(stream_array[0],
                        d_population_new_positive_frequency,
                        d_normalization, population_size, genome_size);
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_deac_dgmmDiv1D(stream_array[1 % MAX_GPU_STREAMS],
                            d_population_new_negative_frequency,
                            d_normalization, population_size, genome_size);
                    #if MAX_GPU_STREAMS > 1
                        GPU_ASSERT(deac_wait(stream_array[1]));
                    #endif
                #endif
                GPU_ASSERT(deac_wait(stream_array[0]));
                gpu_validate_normalization_rows(stream_array[0],
                        d_population_new_positive_frequency,
                        d_normalization, d_normalization_valid, zeroth_moment,
                        population_size, genome_size);
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_validate_normalization_rows(stream_array[1 % MAX_GPU_STREAMS],
                            d_population_new_negative_frequency,
                            d_normalization, d_normalization_valid, zeroth_moment,
                            population_size, genome_size);
                    #if MAX_GPU_STREAMS > 1
                        GPU_ASSERT(deac_wait(stream_array[1]));
                    #endif
                #endif
                GPU_ASSERT(deac_wait(stream_array[0]));
                gpu_cleanup_invalid_normalization_rows(stream_array[0],
                        d_population_new_positive_frequency,
                        d_normalization_valid, population_size, genome_size);
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_cleanup_invalid_normalization_rows(stream_array[0],
                            d_population_new_negative_frequency,
                            d_normalization_valid, population_size, genome_size);
                #endif
                GPU_ASSERT(deac_wait(stream_array[0]));
            #else
                for (size_t i=0; i<population_size; i++) {
                    normalization[i] = 0.0;
                }
                matrix_multiply_MxN_by_Nx1(normalization, population_new_positive_frequency,
                        normalization_term_positive_frequency, population_size, genome_size);
                #ifdef DEAC_TWO_SIDED_POPULATION
                    matrix_multiply_MxN_by_Nx1(normalization, population_new_negative_frequency,
                            normalization_term_negative_frequency, population_size, genome_size);
                #endif
                #ifdef DEAC_TEST_FORCE_INVALID_NORMALIZATION_TRIALS
                    std::fill(
                            normalization,
                            normalization + population_size, 0.0);
                #endif
                for (size_t i=0; i<population_size; i++) {
                    normalization_valid[i] = deac_numerics::try_apply_normalization(
                            zeroth_moment, normalization[i],
                            std::span<double>(
                                    population_new_positive_frequency + i*genome_size,
                                    genome_size)
                            #ifdef DEAC_TWO_SIDED_POPULATION
                                , std::span<double>(
                                        population_new_negative_frequency + i*genome_size,
                                        genome_size)
                            #endif
                            );
                }
            #endif
        }

        #ifdef DEAC_TEST_FORCE_INVALID_NORMALIZATION_TRIALS
            if (normalize) {
                std::cout << "test_forced_invalid_normalization_trials: "
                          << population_size << '\n';
            }
        #endif

        //Rejection
        //Set model isf for new population
        #ifdef USE_GPU
            #ifdef USE_BLAS
                gpu_blas_gemm(default_blas_handle, population_size, number_of_timeslices, genome_size, 1.0, d_population_new_positive_frequency, d_isf_term_positive_frequency, 0.0, d_isf_model);
                GPU_ASSERT(deac_wait(default_stream));
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_blas_gemm(default_blas_handle, population_size, number_of_timeslices, genome_size, 1.0, d_population_new_negative_frequency, d_isf_term_negative_frequency, 1.0, d_isf_model);
                    GPU_ASSERT(deac_wait(default_stream));
                #endif
            #else
                gpu_matmul(default_stream, population_size, number_of_timeslices, genome_size, 1.0, d_population_new_positive_frequency, d_isf_term_positive_frequency, 0.0, d_isf_model);
                GPU_ASSERT(deac_wait(default_stream));
                #ifdef DEAC_TWO_SIDED_POPULATION
                    gpu_matmul(default_stream, population_size, number_of_timeslices, genome_size, 1.0, d_population_new_negative_frequency, d_isf_term_negative_frequency, 1.0, d_isf_model);
                    GPU_ASSERT(deac_wait(default_stream));
                #endif
            #endif
        #else
            for (size_t i=0; i<population_size*number_of_timeslices; i++) {
                isf_model[i] = 0.0;
            }
            deac_numerics::accumulate_population_projection(
                    isf_model, isf_term_positive_frequency, population_new_positive_frequency,
                    number_of_timeslices, genome_size, population_size);
            #ifdef DEAC_TWO_SIDED_POPULATION
                deac_numerics::accumulate_population_projection(
                        isf_model, isf_term_negative_frequency, population_new_negative_frequency,
                        number_of_timeslices, genome_size, population_size);
            #endif
        #endif

        //Set moments
        if (use_negative_first_moment) {
            #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                #ifdef USE_GPU
                    #ifdef USE_BLAS
                        gpu_blas_gemv(default_blas_handle, population_size, number_of_timeslices, 1.0, d_isf_model, d_negative_first_moments_term, 0.0, d_negative_first_moments);
                        GPU_ASSERT(deac_wait(default_stream));
                    #else
                        gpu_deac_gemv(default_stream, population_size, number_of_timeslices, 1.0, d_isf_model, d_negative_first_moments_term, 0.0, d_negative_first_moments);
                        GPU_ASSERT(deac_wait(default_stream));
                    #endif
                #else
                    for (size_t i=0; i<population_size; i++) {
                        negative_first_moments[i] = 0.0;
                    }
                    matrix_multiply_MxN_by_Nx1(negative_first_moments, isf_model,
                            negative_first_moments_term, population_size, number_of_timeslices);
                #endif
            #else
                //FIXME inverse first moment not implemented
            #endif
        }
        if (use_first_moment) {
            #ifdef USE_GPU
                #ifdef USE_BLAS
                    gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0, d_population_new_positive_frequency, d_first_moments_term_positive_frequency, 0.0, d_first_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                    #ifdef DEAC_TWO_SIDED_POPULATION
                        gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0, d_population_new_negative_frequency, d_first_moments_term_negative_frequency, 1.0, d_first_moments);
                        GPU_ASSERT(deac_wait(default_stream));
                    #endif
                #else
                    gpu_deac_gemv(default_stream, population_size, genome_size, 1.0, d_population_new_positive_frequency, d_first_moments_term_positive_frequency, 0.0, d_first_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                    #ifdef DEAC_TWO_SIDED_POPULATION
                        gpu_deac_gemv(default_stream, population_size, genome_size, 1.0, d_population_new_negative_frequency, d_first_moments_term_negative_frequency, 1.0, d_first_moments);
                        GPU_ASSERT(deac_wait(default_stream));
                    #endif
                #endif
            #else
                for (size_t i=0; i<population_size; i++) {
                    first_moments[i] = 0.0;
                }
                matrix_multiply_MxN_by_Nx1(first_moments, population_new_positive_frequency,
                        first_moments_term_positive_frequency, population_size, genome_size);
                #ifdef DEAC_TWO_SIDED_POPULATION
                    matrix_multiply_MxN_by_Nx1(first_moments, population_new_negative_frequency,
                            first_moments_term_negative_frequency, population_size, genome_size);
                #endif
            #endif
        }
        if (use_third_moment) {
            #ifdef USE_GPU
                #ifdef USE_BLAS
                    gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0, d_population_new_positive_frequency, d_third_moments_term_positive_frequency, 0.0, d_third_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                    #ifdef DEAC_TWO_SIDED_POPULATION
                        gpu_blas_gemv(default_blas_handle, population_size, genome_size, 1.0, d_population_new_negative_frequency, d_third_moments_term_negative_frequency, 1.0, d_third_moments);
                        GPU_ASSERT(deac_wait(default_stream));
                    #endif
                #else
                    gpu_deac_gemv(default_stream, population_size, genome_size, 1.0, d_population_new_positive_frequency, d_third_moments_term_positive_frequency, 0.0, d_third_moments);
                    GPU_ASSERT(deac_wait(default_stream));
                    #ifdef DEAC_TWO_SIDED_POPULATION
                        gpu_deac_gemv(default_stream, population_size, genome_size, 1.0, d_population_new_negative_frequency, d_third_moments_term_negative_frequency, 1.0, d_third_moments);
                        GPU_ASSERT(deac_wait(default_stream));
                    #endif
                #endif
            #else
                for (size_t i=0; i<population_size; i++) {
                    third_moments[i] = 0.0;
                }
                matrix_multiply_MxN_by_Nx1(third_moments, population_new_positive_frequency,
                        third_moments_term_positive_frequency, population_size, genome_size);
                #ifdef DEAC_TWO_SIDED_POPULATION
                    matrix_multiply_MxN_by_Nx1(third_moments, population_new_negative_frequency,
                            third_moments_term_negative_frequency, population_size, genome_size);
                #endif
            #endif
        }

        //Set fitness for new population
        #ifdef USE_GPU
            gpu_deac_reduced_chi_squared(default_stream, d_isf_model, d_isf, d_isf_error, d_fitness_new, population_size, number_of_timeslices, 0, 0.0);
            GPU_ASSERT(deac_wait(default_stream));

            #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                if (use_negative_first_moment) {
                    gpu_deac_add_scalar_reduced_chi_squared(default_stream, d_negative_first_moments, negative_first_moment, negative_first_moment_error, d_fitness_new, population_size);
                    GPU_ASSERT(deac_wait(default_stream));
                }
            #else
                //FIXME inverse first moment not implemented
            #endif
            if (use_first_moment) {
                gpu_deac_add_scalar_reduced_chi_squared(default_stream, d_first_moments, first_moment, first_moment_error, d_fitness_new, population_size);
                GPU_ASSERT(deac_wait(default_stream));
            }
            if (use_third_moment) {
                gpu_deac_add_scalar_reduced_chi_squared(default_stream, d_third_moments, third_moment, third_moment_error, d_fitness_new, population_size);
                GPU_ASSERT(deac_wait(default_stream));
            }
            #ifdef DEAC_TEST_POISON_GPU_FITNESS
                test_require_finite_gpu_fitness(d_fitness_new, "evolved");
            #endif
        #else
            // Fitness set in rejection step
        #endif

        //Rejection step
        #ifdef USE_GPU
            size_t grid_size_set_rejection_indices = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            size_t grid_size_swap_control_parameters = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            size_t grid_size_swap_populations = (population_size*genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;

            gpu_set_rejection_indices(default_stream,
                    grid_size_set_rejection_indices, d_rejection_indices,
                    d_fitness_new, d_fitness_old, d_normalization_valid,
                    normalize, population_size);
            GPU_ASSERT(deac_wait(default_stream));
            #ifdef DEAC_TEST_FORCE_INVALID_NORMALIZATION_TRIALS
                GPU_ASSERT(deac_memcpy_device_to_host(
                        fitness_old, d_fitness_new,
                        bytes_fitness, default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                if (!std::all_of(
                        fitness_old, fitness_old + population_size,
                        [](double value) {
                            return value == std::numeric_limits<double>::max();
                        })) {
                    fail_with_error(
                            "forced invalid normalization trial did not "
                            "receive DBL_MAX fitness");
                }
            #endif

            gpu_swap_control_parameters(stream_array[0], grid_size_swap_control_parameters, d_crossover_probabilities_old_positive_frequency, d_crossover_probabilities_new_positive_frequency, d_rejection_indices, population_size);
            gpu_swap_control_parameters(stream_array[1 % MAX_GPU_STREAMS], grid_size_swap_control_parameters, d_differential_weights_old_positive_frequency, d_differential_weights_new_positive_frequency, d_rejection_indices, population_size);
            gpu_swap_populations(stream_array[2 % MAX_GPU_STREAMS], grid_size_swap_populations, d_population_old_positive_frequency, d_population_new_positive_frequency, d_rejection_indices, population_size, genome_size);

            #ifdef DEAC_TWO_SIDED_POPULATION
                gpu_swap_control_parameters(stream_array[3 % MAX_GPU_STREAMS], grid_size_swap_control_parameters, d_crossover_probabilities_old_negative_frequency, d_crossover_probabilities_new_negative_frequency, d_rejection_indices, population_size);
                gpu_swap_control_parameters(stream_array[4 % MAX_GPU_STREAMS], grid_size_swap_control_parameters, d_differential_weights_old_negative_frequency, d_differential_weights_new_negative_frequency, d_rejection_indices, population_size);
                gpu_swap_populations(stream_array[5 % MAX_GPU_STREAMS], grid_size_swap_populations, d_population_old_negative_frequency, d_population_new_negative_frequency, d_rejection_indices, population_size, genome_size);
                #if MAX_GPU_STREAMS > 5
                    GPU_ASSERT(deac_wait(stream_array[5]));
                #endif
                #if MAX_GPU_STREAMS > 4
                    GPU_ASSERT(deac_wait(stream_array[4]));
                #endif
                #if MAX_GPU_STREAMS > 3
                    GPU_ASSERT(deac_wait(stream_array[3]));
                #endif
            #endif
            #if MAX_GPU_STREAMS > 2
                GPU_ASSERT(deac_wait(stream_array[2]));
            #endif
            #if MAX_GPU_STREAMS > 1
                GPU_ASSERT(deac_wait(stream_array[1]));
            #endif
            GPU_ASSERT(deac_wait(stream_array[0]));
        #else
            for (size_t i=0; i<population_size; i++) {
                double _fitness = reduced_chi_square_statistic(isf,
                        isf_model + i*number_of_timeslices, isf_error,
                        number_of_timeslices)/number_of_timeslices;
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    if (use_negative_first_moment) {
                        _fitness += pow((negative_first_moment - negative_first_moments[i])/negative_first_moment_error,2);
                    }
                #else
                    //FIXME inverse first moment not implemented
                #endif
                if (use_first_moment) {
                    _fitness += deac_numerics::scalar_chi_square_penalty(
                            first_moments[i], first_moment, first_moment_error);
                }
                if (use_third_moment) {
                    _fitness += pow((third_moment - third_moments[i])/third_moment_error,2);
                }
                // Rejection step
                if (normalize && !normalization_valid[i]) {
                    _fitness = std::numeric_limits<double>::max();
                }
                #ifdef DEAC_TEST_FORCE_INVALID_NORMALIZATION_TRIALS
                    if (_fitness != std::numeric_limits<double>::max()) {
                        fail_with_error(
                                "forced invalid normalization trial did not "
                                "receive DBL_MAX fitness");
                    }
                #endif
                if ((!normalize || normalization_valid[i])
                        && _fitness <= fitness_old[i]) {
                    fitness_old[i] = _fitness;
                    crossover_probabilities_old_positive_frequency[i] = crossover_probabilities_new_positive_frequency[i];
                    differential_weights_old_positive_frequency[i] = differential_weights_new_positive_frequency[i];
                    #ifdef DEAC_TWO_SIDED_POPULATION
                        crossover_probabilities_old_negative_frequency[i] = crossover_probabilities_new_negative_frequency[i];
                        differential_weights_old_negative_frequency[i] = differential_weights_new_negative_frequency[i];
                    #endif
                    for (size_t j=0; j<genome_size; j++) {
                        population_old_positive_frequency[i*genome_size + j] = population_new_positive_frequency[i*genome_size + j];
                        #ifdef DEAC_TWO_SIDED_POPULATION
                            population_old_negative_frequency[i*genome_size + j] = population_new_negative_frequency[i*genome_size + j];
                        #endif
                    }
                }
            }
        #endif
        #ifdef DEAC_TEST_FORCE_INVALID_NORMALIZATION_TRIALS
            std::cout << "test_invalid_normalization_fitness: DBL_MAX\n";
        #endif
    }

    //Transfer data from gpu to host
    #ifdef USE_GPU
        GPU_ASSERT(deac_memcpy_device_to_host(fitness_old, d_fitness_old, bytes_fitness, stream_array[0]));
        GPU_ASSERT(deac_memcpy_device_to_host(population_old_positive_frequency, d_population_old_positive_frequency, bytes_population, stream_array[1 % MAX_GPU_STREAMS]));
        #ifdef DEAC_TWO_SIDED_POPULATION
            GPU_ASSERT(deac_memcpy_device_to_host(population_old_negative_frequency, d_population_old_negative_frequency, bytes_population, stream_array[1 % MAX_GPU_STREAMS]));
        #endif
        if (track_stats) {
            GPU_ASSERT(deac_memcpy_device_to_host(fitness_mean, d_fitness_mean, bytes_fitness_mean, stream_array[3 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_memcpy_device_to_host(fitness_squared_mean, d_fitness_squared_mean, bytes_fitness_squared_mean, stream_array[4 % MAX_GPU_STREAMS]));
        }
        for (auto& s : stream_array) {
            GPU_ASSERT(deac_wait(s));
        }
    #endif

    std::tie(minimum_fitness_idx, minimum_fitness) = argmin_and_min(fitness_old, population_size);

    double * best_dsf;
    double * best_frequency;
    #ifdef DEAC_TWO_SIDED_POPULATION
        best_dsf = (double*) malloc(sizeof(double)*(2*genome_size - 1));
        best_frequency = (double*) malloc(sizeof(double)*(2*genome_size - 1));
    #else
        best_dsf = (double*) malloc(sizeof(double)*genome_size);
        best_frequency = (double*) malloc(sizeof(double)*genome_size);
    #endif
    for (size_t i=0; i<genome_size; i++) {
        double f = frequency[i];
        #ifdef DEAC_TWO_SIDED_POPULATION
            size_t idx_p = genome_size + i - 1;
            size_t idx_n = genome_size - i - 1;
        #else
            size_t idx_p = i;
        #endif
        best_frequency[idx_p] = f;
        #ifdef DEAC_TWO_SIDED_POPULATION
            best_frequency[idx_n] = -f;
        #endif
        #ifdef USE_GPU
            size_t idx_dsf = population_size*i + minimum_fitness_idx; // Column-major storage
        #else
            size_t idx_dsf = genome_size*minimum_fitness_idx + i; // Row-major storage
        #endif
        #ifndef ZEROT
            #ifdef USE_HYPERBOLIC_MODEL
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                     best_dsf[idx_p] = 0.5*population_old_positive_frequency[idx_dsf]*exp(0.5*beta*f);
                #else
                    if (spectra_type == "spbsf") {
                        best_dsf[idx_p] = -2.0*M_PI*population_old_positive_frequency[idx_dsf]*sinh(beta*f/2); // 0.5*exp(0.5*beta*f)*(-2.0*M_PI*(1 - exp(-beta*f)))
                        best_dsf[idx_n] = 2.0*M_PI*population_old_negative_frequency[idx_dsf]*sinh(beta*f/2);  // 0.5*exp(-0.5*beta*f)*(-2.0*M_PI*(1 - exp(beta*f))) <-- f is negative here
                    } else if (spectra_type == "spfsf") {
                        best_dsf[idx_p] = -2.0*M_PI*population_old_positive_frequency[idx_dsf]*cosh(beta*f/2); // 0.5*exp(0.5*beta*f)*(-2.0*M_PI*(1 + exp(-beta*f)))
                        best_dsf[idx_n] = -2.0*M_PI*population_old_negative_frequency[idx_dsf]*cosh(beta*f/2);  // 0.5*exp(-0.5*beta*f)*(-2.0*M_PI*(1 + exp(beta*f))) <-- f is negative here
                    } else {
                        best_dsf[idx_p] = 0.5*population_old_positive_frequency[idx_dsf]*exp(0.5*beta*f);
                        best_dsf[idx_n] = 0.5*population_old_negative_frequency[idx_dsf]*exp(-0.5*beta*f);
                    }
                #endif
            #endif
            #ifdef USE_STANDARD_MODEL
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    best_dsf[idx_p] = population_old_positive_frequency[idx_dsf];
                #else
                    if (spectra_type == "spbsf") {
                        best_dsf[idx_p] = -2.0*M_PI*population_old_positive_frequency[idx_dsf]*(1.0 - exp(-beta*f)); // *(-2.0*M_PI*(1 - exp(-beta*f)))
                        best_dsf[idx_n] = -2.0*M_PI*population_old_negative_frequency[idx_dsf]*(1.0 - exp(beta*f));  // *(-2.0*M_PI*(1 - exp(beta*f))) <-- f is negative here
                    } else if (spectra_type == "spfsf") {
                        best_dsf[idx_p] = -2.0*M_PI*population_old_positive_frequency[idx_dsf]*(1.0 + exp(-beta*f)); // *(-2.0*M_PI*(1 + exp(-beta*f)))
                        best_dsf[idx_n] = -2.0*M_PI*population_old_negative_frequency[idx_dsf]*(1.0 + exp(beta*f));  // *(-2.0*M_PI*(1 + exp(beta*f))) <-- f is negative here
                    } else {
                        best_dsf[idx_p] = population_old_positive_frequency[idx_dsf];
                        best_dsf[idx_n] = population_old_negative_frequency[idx_dsf];
                    }
                #endif
            #endif
            #ifdef USE_NORMALIZATION_MODEL
                #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    best_dsf[idx_p] = population_old_positive_frequency[idx_dsf]/(1.0 + exp(-beta*f));
                #else
                    if (spectra_type == "spbsf") {
                        best_dsf[idx_p] = -2.0*M_PI*population_old_positive_frequency[idx_dsf]*tanh(0.5*beta*f); // (1/(1 + exp(-beta*f)))*(-2.0*M_PI*(1 - exp(-beta*f)))
                        best_dsf[idx_n] = 2.0*M_PI*population_old_negative_frequency[idx_dsf]*tanh(0.5*beta*f);  // (1/(1 + exp(beta*f)))*(-2.0*M_PI*(1 - exp(beta*f))) <-- f is negative here
                    } else if (spectra_type == "spfsf") {
                        best_dsf[idx_p] = -2.0*M_PI*population_old_positive_frequency[idx_dsf]; // (1/(1 + exp(-beta*f)))*(-2.0*M_PI*(1 + exp(-beta*f)))
                        best_dsf[idx_n] = -2.0*M_PI*population_old_negative_frequency[idx_dsf];  // (1/(1 + exp(-beta*f)))*(-2.0*M_PI*(1 + exp(beta*f))) <-- f is negative here
                    } else {
                        best_dsf[idx_p] = population_old_positive_frequency[idx_dsf]/(1.0 + exp(-beta*f));
                        best_dsf[idx_n] = population_old_negative_frequency[idx_dsf]/(1.0 + exp(beta*f));
                    }
                #endif
            #endif
        #else
            best_dsf[idx_p] = population_old_positive_frequency[idx_dsf];
            #ifdef DEAC_TWO_SIDED_POPULATION
                best_dsf[idx_n] = population_old_negative_frequency[idx_dsf];
            #endif
        #endif
    }

    //Get Statistics
    if (generation == number_of_generations - 2) {
        generation += 1;
        if (track_stats) {
            fitness_mean[generation] = mean(fitness_old, population_size);
            fitness_minimum[generation] = minimum_fitness;
            fitness_squared_mean[generation] = squared_mean(fitness_old, population_size);
        }
    }

    std::exception_ptr result_io_error;
    try {
        // Save data.
        #ifndef ZEROT
            std::string deac_prefix = "deac-" + spectra_type;
        #else
            std::string deac_prefix = "deac-zT";
        #endif
        std::string best_dsf_filename_str = string_format("%s_dsf_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
        fs::path best_dsf_filename = save_directory / best_dsf_filename_str;
        std::string frequency_filename_str = string_format("%s_frequency_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
        fs::path frequency_filename = save_directory / frequency_filename_str;
        #ifdef DEAC_TWO_SIDED_POPULATION
            const size_t output_size = 2*genome_size - 1;
            deac_io::write_binary_doubles(
                    best_dsf_filename, std::span<const double>(best_dsf, output_size));
            deac_io::write_binary_doubles(
                    frequency_filename, std::span<const double>(best_frequency, output_size));
        #else
            deac_io::write_binary_doubles(
                    best_dsf_filename, std::span<const double>(best_dsf, genome_size));
            deac_io::write_binary_doubles(
                    frequency_filename, std::span<const double>(best_frequency, genome_size));
        #endif
        fs::path fitness_mean_filename;
        fs::path fitness_minimum_filename;
        fs::path fitness_squared_mean_filename;
        if (track_stats) {
            std::string fitness_mean_filename_str = string_format("%s_stats_fitness-mean_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
            std::string fitness_minimum_filename_str = string_format("%s_stats_fitness-minimum_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
            std::string fitness_squared_mean_filename_str = string_format("%s_stats_fitness-squared-mean_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
            fitness_mean_filename = save_directory / fitness_mean_filename_str;
            fitness_minimum_filename = save_directory / fitness_minimum_filename_str;
            fitness_squared_mean_filename = save_directory / fitness_squared_mean_filename_str;
            deac_io::write_binary_doubles(
                    fitness_mean_filename,
                    std::span<const double>(fitness_mean, generation + 1));
            deac_io::write_binary_doubles(
                    fitness_minimum_filename,
                    std::span<const double>(fitness_minimum, generation + 1));
            deac_io::write_binary_doubles(
                    fitness_squared_mean_filename,
                    std::span<const double>(fitness_squared_mean, generation + 1));
        }
        // Append the final run record only after all binary artifacts are
        // successfully flushed and closed.
        std::string log_filename_str = string_format("%s_log_%s.dat",deac_prefix.c_str(),uuid_str.c_str());
        fs::path log_filename = save_directory / log_filename_str;
        std::ostringstream log;

        // Build Type
        #ifdef USE_HYPERBOLIC_MODEL
            log << "build: USE_HYPERBOLIC_MODEL" << std::endl;
        #endif
        #ifdef USE_STANDARD_MODEL
            log << "build: USE_STANDARD_MODEL" << std::endl;
        #endif
        #ifdef USE_NORMALIZATION_MODEL
            log << "build: USE_NORMALIZATION_MODEL" << std::endl;
        #endif
        #ifndef ZEROT
            log << "kernel: " << spectra_type << std::endl;
        #else
            log << "kernel: zero-temperature-positive-laplace" << std::endl;
        #endif

        // Input parameters
        log << "temperature: " << temperature << std::endl;
        log << "number_of_generations: " << number_of_generations << std::endl;
        log << "number_of_timeslices: " << number_of_timeslices << std::endl;
        log << "population_size: " << population_size << std::endl;
        log << "genome_size: " << genome_size << std::endl;
        log << "normalize: " << normalize << std::endl;
        log << "use_negative_first_moment: " << use_negative_first_moment << std::endl;
        #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            if (use_negative_first_moment) {
                log << "negative_first_moment: " << negative_first_moment << std::endl;
                log << "negative_first_moment_error: " << negative_first_moment_error << std::endl;
            }
        #endif
        log << "first_moment: " << first_moment << std::endl;
        if (use_first_moment) {
            log << "first_moment_error: " << first_moment_error << std::endl;
        }
        log << "third_moment: " << third_moment << std::endl;
        log << "third_moment_error: " << third_moment_error << std::endl;
        log << "crossover_probability: " << crossover_probability << std::endl;
        log << "self_adapting_crossover_probability: " << self_adapting_crossover_probability << std::endl;
        log << "differential_weight: " << differential_weight << std::endl;
        log << "self_adapting_differential_weight_probability: " << self_adapting_differential_weight_probability << std::endl;
        log << "stop_minimum_fitness: " << stop_minimum_fitness << std::endl;
        log << "track_stats: " << track_stats << std::endl;
        log << "seed: " << seed << std::endl;

        // Generated variables
        log << "best_dsf_filename: " << best_dsf_filename << std::endl;
        log << "frequency_filename: " << frequency_filename << std::endl;
        log << "generation: " << generation << std::endl;
        log << "minimum_fitness: " << minimum_fitness << std::endl;
        if (track_stats) {
            log << "fitness_mean_filename: " << fitness_mean_filename << std::endl;
            log << "fitness_minimum_filename: " << fitness_minimum_filename << std::endl;
            log << "fitness_squared_mean_filename: " << fitness_squared_mean_filename << std::endl;
        }
        deac_io::append_text(log_filename, log.str());
        std::cout << "generation: " << generation << std::endl;
        std::cout << "minimum_fitness: " << minimum_fitness << std::endl;
    } catch (...) {
        // Keep the original I/O failure while still releasing all solver resources.
        result_io_error = std::current_exception();
    }

    //Free memory
    if (track_stats) {
        free(fitness_mean);
        free(fitness_minimum);
        free(fitness_squared_mean);
    }

    free(fitness_old);
    free(mutant_indices);
    free(best_dsf);
    free(best_frequency);

    free(isf_term_positive_frequency);
    free(population_old_positive_frequency);
    free(population_new_positive_frequency);
    if (normalize) {
        free(normalization);
        free(normalization_valid);
        free(normalization_term_positive_frequency);
    }
    if (use_first_moment) {
        free(first_moments);
        free(first_moments_term_positive_frequency);
    }
    if (use_third_moment) {
        free(third_moments);
        free(third_moments_term_positive_frequency);
    }
    free(isf_model);
    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        //FIXME need to add inverse first moment functionality then can remove this ifdef
        if (use_negative_first_moment) {
            free(negative_first_moments_term);
            free(negative_first_moments);
        }
    #endif
    free(crossover_probabilities_old_positive_frequency);
    free(crossover_probabilities_new_positive_frequency);
    free(differential_weights_old_positive_frequency);
    free(differential_weights_new_positive_frequency);
    free(mutate_indices_positive_frequency);

    #ifdef DEAC_TWO_SIDED_POPULATION
        free(isf_term_negative_frequency);
        free(population_old_negative_frequency);
        free(population_new_negative_frequency);
        if (normalize) {
            free(normalization_term_negative_frequency);
        }
        if (use_first_moment) {
            free(first_moments_term_negative_frequency);
        }
        if (use_third_moment) {
            free(third_moments_term_negative_frequency);
        }
        //FIXME need to add inverse first moment functionality
        //if (use_negative_first_moment) {
        //    free(negative_first_moments_term_negative_frequency);
        //}
        free(crossover_probabilities_old_negative_frequency);
        free(crossover_probabilities_new_negative_frequency);
        free(differential_weights_old_negative_frequency);
        free(differential_weights_new_negative_frequency);
        free(mutate_indices_negative_frequency);
    #endif

    #ifdef USE_GPU
        free(rng_state);

        // Release device memory
        if (track_stats) {
            GPU_ASSERT(deac_free(d_fitness_mean,         stream_array[0 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_fitness_squared_mean, stream_array[1 % MAX_GPU_STREAMS]));
        }

        GPU_ASSERT(deac_free(d_isf,               stream_array[2 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_isf_error,         stream_array[3 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_rejection_indices, stream_array[4 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_mutant_indices,    stream_array[5 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_minimum_fitness,   stream_array[6 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_rng_state,         stream_array[7 % MAX_GPU_STREAMS]));

        GPU_ASSERT(deac_free(d_isf_term_positive_frequency,       stream_array[ 8 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_population_old_positive_frequency, stream_array[ 9 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_population_new_positive_frequency, stream_array[10 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_fitness_old,                       stream_array[11 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_fitness_new,                       stream_array[12 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_isf_model,                         stream_array[13 % MAX_GPU_STREAMS]));
        if (normalize) {
            GPU_ASSERT(deac_free(d_normalization,                         stream_array[14 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_normalization_valid,                   stream_array[15 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_normalization_term_positive_frequency, stream_array[16 % MAX_GPU_STREAMS]));
        }
        if (use_first_moment) {
            GPU_ASSERT(deac_free(d_first_moments,                         stream_array[16 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_first_moments_term_positive_frequency, stream_array[17 % MAX_GPU_STREAMS]));
        }
        if (use_third_moment) {
            GPU_ASSERT(deac_free(d_third_moments,                         stream_array[18 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_third_moments_term_positive_frequency, stream_array[19 % MAX_GPU_STREAMS]));
        }
        #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            //FIXME need to add inverse first moment functionality, then can remove this ifdef
            if (use_negative_first_moment) {
                GPU_ASSERT(deac_free(d_negative_first_moments,                         stream_array[20 % MAX_GPU_STREAMS]));
                GPU_ASSERT(deac_free(d_negative_first_moments_term, stream_array[21 % MAX_GPU_STREAMS]));
            }
        #endif
        GPU_ASSERT(deac_free(d_crossover_probabilities_old_positive_frequency, stream_array[22 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_crossover_probabilities_new_positive_frequency, stream_array[23 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_differential_weights_old_positive_frequency,    stream_array[24 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_differential_weights_new_positive_frequency,    stream_array[25 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_mutate_indices_positive_frequency,              stream_array[26 % MAX_GPU_STREAMS]));

        #ifdef DEAC_TWO_SIDED_POPULATION
            GPU_ASSERT(deac_free(d_isf_term_negative_frequency,       stream_array[27 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_population_old_negative_frequency, stream_array[28 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_population_new_negative_frequency, stream_array[29 % MAX_GPU_STREAMS]));
            if (normalize) {
                GPU_ASSERT(deac_free(d_normalization_term_negative_frequency, stream_array[30 % MAX_GPU_STREAMS]));
            }
            if (use_first_moment) {
                GPU_ASSERT(deac_free(d_first_moments_term_negative_frequency, stream_array[31 % MAX_GPU_STREAMS]));
            }
            if (use_third_moment) {
                GPU_ASSERT(deac_free(d_third_moments_term_negative_frequency, stream_array[32 % MAX_GPU_STREAMS]));
            }
            //FIXME need to add inverse first moment functionality
            //if (use_negative_first_moment) {
            //    GPU_ASSERT(deac_free(d_negative_first_moments_term_negative_frequency, stream_array[33 % MAX_GPU_STREAMS]));
            //}
            GPU_ASSERT(deac_free(d_crossover_probabilities_old_negative_frequency, stream_array[34 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_crossover_probabilities_new_negative_frequency, stream_array[35 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_differential_weights_old_negative_frequency,    stream_array[36 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_differential_weights_new_negative_frequency,    stream_array[37 % MAX_GPU_STREAMS]));
            GPU_ASSERT(deac_free(d_mutate_indices_negative_frequency,              stream_array[38 % MAX_GPU_STREAMS]));
        #endif

        for (auto& s : stream_array) {
            GPU_ASSERT(deac_wait(s));
        }

        // Destroy Streams
        for (size_t i = 0; i < MAX_GPU_STREAMS; i++) {
            GPU_ASSERT(deac_stream_destroy(stream_array[i]));
        }

        #ifdef USE_BLAS
            //Destroy GPU device BLAS handles
            for (size_t i = 0; i < MAX_GPU_STREAMS; i++) {
                GPU_BLAS_ASSERT(deac_destroy_blas_handle(blas_handle_array[i]));
            }
        #endif
    #endif

    if (result_io_error) {
        std::rethrow_exception(result_io_error);
    }
}

int deac_main (int argc, char *argv[]) {
    argparse::ArgumentParser program(
        "deac-cpp", std::string(deac_build_identity::semantic_version()));
    program.add_argument("--build-identity")
        .action([](const auto&) {
            print_build_identity(std::cout);
            std::exit(0);
        })
        .default_value(false)
        .help("prints canonical JSON build identity and exits")
        .implicit_value(true)
        .nargs(0);
    program.add_argument("--build-receipt")
        .action([](const auto&) {
            print_build_receipt(std::cout);
            std::exit(0);
        })
        .default_value(false)
        .help("prints canonical JSON effective build receipt and exits")
        .implicit_value(true)
        .nargs(0);
    program.add_argument("-T", "--temperature")
        #ifndef ZEROT
            .help("Temperature of system. Must be positive.")
        #else
            .help("Ignored by ZeroT builds; the temperature is fixed to zero.")
        #endif
        .default_value(0.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-N", "--number_of_generations")
        .help("Number of generations before genetic algorithm quits.")
        .default_value(100000UL)
        .action([](const std::string& value) { return std::stoul(value); });
    program.add_argument("-P","--population_size")
        .help("Size of initial population")
        .default_value(512UL)
        .action([](const std::string& value) { return std::stoul(value); });
    program.add_argument("-M","--genome_size")
        .help("Size of genome.")
        .default_value(512UL)
        .action([](const std::string& value) { return std::stoul(value); });
    program.add_argument("--omega_max")
        .help("Maximum frequency to explore.")
        .default_value(60.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--frequency_file")
        .help("Filename containing frequency partition (genome_size and omega_max will be ignored).");
    program.add_argument("--normalize")
        .help("Normalize spectrum to the zeroth moment from the first ISF "
              "value, which must be finite, positive, normal, and representable "
              "for the selected model and frequency grid.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--use_negative_first_moment")
        .help("Calculate inverse first moment from ISF data and use it in fitness.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--first_moment")
        .help("Set first frequency moment and use in fitness function.")
        .default_value(-1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--first_moment_error")
        .help("Set the finite, positive standard deviation for an active "
              "first frequency moment. Defaults to 1.0 and must not be "
              "supplied when first_moment is negative.")
        .default_value(1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--third_moment")
        .help("Set third frequency moment and use in fitness function.")
        .default_value(-1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--third_moment_error")
        .help("Set error for third frequency moment.")
        .default_value(0.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-r","--crossover_probability")
        .help("Initial probability for parent gene to become mutant vector gene. Must be finite and in [0, 1].")
        .default_value(0.9)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-u","--self_adapting_crossover_probability")
        .help("Probability for `crossover_probability` to mutate. Must be finite and in [0, 1].")
        .default_value(0.1)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-F","--differential_weight")
        .help("Initial weight factor when creating mutant vector. Must be finite and in [0, 2].")
        .default_value(0.9)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--self_adapting_differential_weight_probability")
        .help("Probability for `differential_weight` to mutate. Must be finite and in [0, 1].")
        .default_value(0.1)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--stop_minimum_fitness")
        .help("Stop evolving when minimum fitness is at or below this finite value. Negative values are allowed.")
        .default_value(1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--seed")
        .help("Seed to pass to random number generator.")
        .default_value(0UL)
        .action([](const std::string& value) { return std::stoul(value); });
    program.add_argument("--uuid")
        .help("UUID for run. If empty will be set to `seed`.")
        .default_value("");
    program.add_argument("--save_directory")
        .help("Directory to save results in.")
        .default_value("./deacresults");
    program.add_argument("--track_stats")
        .help("Track minimum fitness and other stats.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--spectra_type")
        #ifdef ZEROT
            .help("ZeroT uses the fixed one-sided positive-frequency Laplace kernel [positive].")
            .default_value("positive");
        #elif defined(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF)
            .help("Choose spectral type for kernel factors [bdsf].")
            .default_value("bdsf");
        #else
            .help("Choose spectral type for kernel factors [spbsf, spfsf, bfull, ffull].")
            .default_value("spfsf");
        #endif
    program.add_argument("isf_file") //FIXME make this more generic
        .help("binary file containing isf data (tau, isf, error)");
    try {
      program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program << std::endl;
        return 1;
    }

    double crossover_probability = program.get<double>("--crossover_probability");
    double self_adapting_crossover_probability = program.get<double>("--self_adapting_crossover_probability");
    double differential_weight = program.get<double>("--differential_weight");
    double self_adapting_differential_weight_probability = program.get<double>("--self_adapting_differential_weight_probability");
    double stop_minimum_fitness = program.get<double>("--stop_minimum_fitness");
    try {
        deac_configuration::validate_evolution_controls({
                crossover_probability,
                self_adapting_crossover_probability,
                differential_weight,
                self_adapting_differential_weight_probability,
                stop_minimum_fitness});
    } catch (const std::invalid_argument& error) {
        fail_with_error(error.what());
    }

    double first_moment = program.get<double>("--first_moment");
    double first_moment_error = program.get<double>("--first_moment_error");
    if (!std::isfinite(first_moment)) {
        fail_with_error("first_moment must be finite");
    }
    if (first_moment >= 0.0
            && (!std::isfinite(first_moment_error)
                || first_moment_error <= 0.0)) {
        fail_with_error(
                "first_moment_error must be finite and positive when "
                "first_moment is used");
    }
    if (first_moment < 0.0 && program.is_used("--first_moment_error")) {
        fail_with_error(
                "--first_moment_error requires an active --first_moment");
    }

    std::string uuid_str = program.get<std::string>("--uuid");
    if (uuid_str == "") {
        uuid_str = std::to_string(program.get<unsigned long>("--seed"));
    }

    std::string spectra_type = program.get<std::string>("--spectra_type");
    #ifdef ZEROT
        std::string valid_spectra_type = "positive";
    #elif defined(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF)
        std::string valid_spectra_type = "bdsf";
    #else
        std::string valid_spectra_type = "spbsf, spfsf, bfull, ffull";
    #endif
    if (!(
         #ifdef ZEROT
             (spectra_type == "positive")
         #elif defined(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF)
             (spectra_type == "bdsf")
         #else
             (spectra_type == "spbsf") ||
             (spectra_type == "spfsf") ||
             (spectra_type == "bfull") ||
             (spectra_type == "ffull")
         #endif
        )) {
        std::cout << "Please choose spectra_type from the following options: " << valid_spectra_type << std::endl;
        return 1;
    }

    std::string isf_file = program.get<std::string>("isf_file");
    std::vector<double> numpy_data = deac_io::read_binary_doubles(isf_file);
    const size_t number_of_elements = numpy_data.size();
    if (number_of_elements % 3 != 0) {
        fail_with_error("ISF input file must contain tau, isf, and error arrays of equal length");
    }
    size_t number_of_timeslices = number_of_elements/3;
    if (number_of_timeslices < 2) {
        fail_with_error("ISF input file must contain at least two timeslices");
    }

    double * const imaginary_time = numpy_data.data();
    double * const isf = numpy_data.data() + number_of_timeslices;
    double * const isf_error = numpy_data.data() + 2*number_of_timeslices;
    bool normalize = program.get<bool>("--normalize");
    if (normalize) {
        deac_numerics::validate_normalization_target(isf[0]);
        #ifdef ALLOW_NEGATIVE_SPECTRAL_WEIGHT
            fail_with_error(
                    "--normalize is incompatible with negative spectral weights");
        #endif
    }
    for (size_t i=0; i<number_of_timeslices; i++) {
        if (!std::isfinite(imaginary_time[i]) || !std::isfinite(isf[i]) || !std::isfinite(isf_error[i])) {
            fail_with_error("ISF input file contains non-finite values");
        }
        #ifdef DEAC_TWO_SIDED_POPULATION
            if (((spectra_type == "spbsf") || (spectra_type == "spfsf")) && (isf[i] > 0.0)) {
                fail_with_error("positive ISF values are not supported for single-particle spectra");
            }
        #endif
        if (isf_error[i] <= 0.0) {
            fail_with_error("ISF input errors must be positive");
        }
    }

    uint64_t seed = 1407513600 + static_cast<uint64_t>(program.get<unsigned long>("--seed"));
    uint64_t seed_int = static_cast<uint64_t>(program.get<unsigned long>("--seed"));
    struct xoshiro256p_state rng = xoshiro256p_init(seed);

    double temperature = program.get<double>("--temperature");
    #ifndef ZEROT
        if (!std::isfinite(temperature) || temperature <= 0.0) {
            fail_with_error("temperature must be positive for non-ZeroT builds");
        }
    #endif
    size_t number_of_generations = static_cast<size_t>(program.get<unsigned long>("--number_of_generations"));
    size_t population_size = static_cast<size_t>(program.get<unsigned long>("--population_size"));
    if (number_of_generations < 2) {
        fail_with_error("number_of_generations must be at least 2");
    }
    if (population_size < 4) {
        fail_with_error("population_size must be at least 4");
    }

    size_t genome_size;
    std::vector<double> frequency_data;
    if (auto frequency_filename = program.present("--frequency_file")) {
        frequency_data = deac_io::read_binary_doubles(*frequency_filename);
        genome_size = frequency_data.size();
    } else{
        genome_size = static_cast<size_t>(program.get<unsigned long>("--genome_size"));
        double max_frequency = program.get<double>("--omega_max");
        if (genome_size < 2) {
            fail_with_error("genome_size must be at least 2");
        }
        if (!std::isfinite(max_frequency) || max_frequency <= 0.0) {
            fail_with_error("omega_max must be positive");
        }

        frequency_data.resize(genome_size);
        double dfrequency = max_frequency/(genome_size - 1);
        for (size_t i=0; i<genome_size; i++) {
            frequency_data[i] = i*dfrequency;
        }
    }
    double* const frequency = frequency_data.data();
    if (genome_size < 2) {
        fail_with_error("frequency_file must contain at least two frequencies");
    }
    for (size_t i=0; i<genome_size; i++) {
        if (!std::isfinite(frequency[i]) || frequency[i] < 0.0) {
            fail_with_error("frequencies must be finite and non-negative");
        }
        if ((i > 0) && (frequency[i] <= frequency[i - 1])) {
            fail_with_error("frequencies must be strictly increasing");
        }
    }
    if (normalize) {
        const std::vector<double> frequency_weights =
                deac_numerics::trapezoidal_weights(frequency_data);
        const deac_numerics::NormalizationTerms normalization_terms =
                deac_numerics::make_normalization_terms(
                        frequency_data, frequency_weights, temperature);
        deac_numerics::validate_initial_normalization_scale(
                isf[0], normalization_terms.maximum_initial_denominator);
    }

    bool use_negative_first_moment = program.get<bool>("--use_negative_first_moment");
    #if !defined(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF) || defined(ZEROT)
        if (use_negative_first_moment) {
            fail_with_error(
                    "use_negative_first_moment requires a finite-temperature "
                    "bosonic detailed-balance build");
        }
    #endif
    double third_moment = program.get<double>("--third_moment");
    double third_moment_error = program.get<double>("--third_moment_error");
    if (!std::isfinite(third_moment)) {
        fail_with_error("third_moment must be finite");
    }
    if ((third_moment >= 0.0) && (!std::isfinite(third_moment_error) || third_moment_error <= 0.0)) {
        fail_with_error("third_moment_error must be positive when third_moment is used");
    }

    bool track_stats = program.get<bool>("--track_stats");
    std::string save_directory_str = program.get<std::string>("--save_directory");
    fs::path save_directory(save_directory_str);
    std::cout << "uuid: " << uuid_str << std::endl;
    deac_io::ensure_result_directory(save_directory);

    //Write to log file
    #ifndef ZEROT
        std::string deac_prefix = "deac-" + spectra_type;
    #else
        std::string deac_prefix = "deac-zT";
    #endif
    std::string log_filename_str = string_format("%s_log_%s.dat",deac_prefix.c_str(),uuid_str.c_str());
    fs::path log_filename = save_directory / log_filename_str;
    std::ostringstream initial_log;
    initial_log << "uuid: " << uuid_str << std::endl;
    initial_log << "isf_file: " << isf_file << std::endl;
    deac_io::append_text(log_filename, initial_log.str());

    deac( &rng, imaginary_time, isf, isf_error, frequency, temperature,
            number_of_generations, number_of_timeslices, population_size, genome_size,
            normalize, use_negative_first_moment, first_moment, first_moment_error,
            third_moment, third_moment_error, crossover_probability,
            self_adapting_crossover_probability, differential_weight,
            self_adapting_differential_weight_probability,
            stop_minimum_fitness,
            track_stats, seed_int, uuid_str, spectra_type, save_directory);

    return 0;
}

int main (int argc, char *argv[]) {
    try {
        return deac_main(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
