#include <stdio.h>
#include <math.h> // cosh
#include <stdlib.h>
#include <iostream>
#include <tuple> // for tie() and tuple
#include <argparse.hpp>
#include <sstream> // 
#include <algorithm> // std::none_of
#include <rng.hpp>
#include <memory> // string_format
#include <string> // string_format
#include <stdexcept> // throw
#include <fstream> // std::ofstream
#include <cassert>
#include <uuid.h>
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

#ifdef DEAC_DEBUG
    #include "deac_debug.h"
#endif

void write_to_logfile(fs::path filename, std::string log_message ) {
    std::ofstream ofs(filename.c_str(), std::ios_base::out | std::ios_base::app );
    ofs << log_message << std::endl;
    ofs.close();
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

void write_array(fs::path filename, double * buffer, size_t length) {
    FILE * output_file;
    output_file = fopen (filename.c_str(), "wb");
    fwrite (buffer , sizeof(double), static_cast<size_t>(length), output_file);
    fclose (output_file);
}

std::tuple <double*, size_t> load_numpy_array(std::string data_file) {
    FILE * input_file;
    long file_size_bytes;
    double * buffer;
    size_t result;
  
    input_file = fopen( data_file.c_str(), "rb" );
    if (input_file==NULL) {
        std::string error_str = string_format("File error: %s\n", data_file.c_str());
        fputs(error_str.c_str(), stderr);
        exit(1);
    }
  
    // obtain file size:
    fseek(input_file , 0 , SEEK_END);
    file_size_bytes = ftell(input_file);
    rewind(input_file);
    
    size_t number_of_elements = static_cast<size_t> (file_size_bytes/sizeof(double));
  
    // allocate memory to contain the whole file:
    buffer = (double*) malloc(sizeof(char)*file_size_bytes);
    if (buffer == NULL) {fputs("Memory error",stderr); exit(2);}
  
    // copy the file into the buffer:
    result = fread(buffer,1,file_size_bytes,input_file);
    if (result != file_size_bytes) {fputs("Reading error",stderr); exit(3);}
  
    /* the whole file is now loaded in the memory buffer. */
    fclose (input_file);

    std::tuple <double*, size_t> numpy_data_tuple(buffer, number_of_elements);
    return numpy_data_tuple;
}

void matrix_multiply_MxN_by_Nx1(double * C, double * A, double * B, size_t M, size_t N) {
    for (size_t i=0; i<M; i++) {
        for (size_t j=0; j<N; j++) {
            C[i] += A[i*N + j]*B[j];
        }
    }
}

void matrix_multiply_LxM_by_MxN(double * C, double * A, double * B, size_t L, size_t M, size_t N) {
    for (size_t i=0; i<N; i++) {
        for (size_t j=0; j<L; j++) {
            for (size_t k=0; k<M; k++) {
                C[i*L + j] += A[j*M + k]*B[i*M + k];
            }
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

size_t argmin(double * A, size_t length) {
    size_t _argmin=0;
    double _minimum = A[0];
    for (size_t i=0; i<length; i++) {
        if (A[i] < _minimum) {
            _minimum = A[i];
            _argmin = i;
        }
    }
    return _argmin;
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

std::tuple <size_t, size_t, size_t> get_mutant_indices(struct xoshiro256p_state * rng, size_t mutant_index0, size_t length) {
    size_t mutant_index1 = mutant_index0;
    size_t mutant_index2 = mutant_index0;
    size_t mutant_index3 = mutant_index0;
    while (mutant_index1 == mutant_index0) {
        mutant_index1 = xoshiro256p(rng) % length;
    }

    while ((mutant_index2 == mutant_index0) || (mutant_index2 == mutant_index1)) {
        mutant_index2 = xoshiro256p(rng) % length;
    }

    while ((mutant_index3 == mutant_index0) || (mutant_index3 == mutant_index1)
            || (mutant_index3 == mutant_index2)) {
        mutant_index3 = xoshiro256p(rng) % length;
    }

    std::tuple <size_t, size_t, size_t> _mutant_indices(mutant_index1, mutant_index2, mutant_index3);
    return _mutant_indices;
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

void deac(struct xoshiro256p_state * rng, double * const imaginary_time,
        double * const isf, double * const isf_error, double * frequency,
        double temperature, size_t number_of_generations, size_t number_of_timeslices, size_t population_size,
        size_t genome_size, bool normalize, bool use_negative_first_moment, 
        double first_moment, double third_moment, double third_moment_error,
        double crossover_probability,
        double self_adapting_crossover_probability,
        double differential_weight, 
        double self_adapting_differential_weight_probability,
        double self_adapting_differential_weight_shift,
        double self_adapting_differential_weight, double stop_minimum_fitness,
        bool track_stats, size_t seed, std::string uuid_str, std::string spectra_type, fs::path save_directory) {

    #ifdef ZEROT
        //Set flags and temperature
        if (use_negative_first_moment) {
            std::cout << "use_negative_first_moment disabled for zero temperature build" << std::endl;
        }
        use_negative_first_moment = false; //FIXME disabling inverse first moment for zero temperature (needs further investigation)
        temperature = 0.0;
    #endif

    #ifdef USE_GPU
        //Create GPU device streams
        deac_stream_t stream_array[MAX_GPU_STREAMS];
        for (size_t i = 0; i < MAX_GPU_STREAMS; i++) {
            GPU_ASSERT(deac_stream_create(stream_array[i]));
        }
        
        // Set up default "stream"
        auto default_stream = stream_array[0];

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

    //Set isf term for trapezoidal rule integration with dsf (population members)
    size_t bytes_isf_term = sizeof(double)*genome_size*number_of_timeslices;
    double * isf_term_positive_frequency;
    isf_term_positive_frequency = (double*) malloc(bytes_isf_term);
    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        double * isf_term_negative_frequency;
        isf_term_negative_frequency = (double*) malloc(bytes_isf_term);
    #endif
    for (size_t i=0; i<number_of_timeslices; i++) {
        double t = imaginary_time[i];
        #ifndef ZEROT
            #ifdef USE_HYPERBOLIC_MODEL
                double bo2mt = 0.5*beta - t;
            #endif
            #ifdef USE_STANDARD_MODEL
                double bmt = beta - t;
            #endif
            #ifdef USE_NORMALIZATION_MODEL
                double bmt = beta - t;
            #endif
        #endif
        for (size_t j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            size_t isf_term_idx = i*genome_size + j;
            #ifndef ZEROT
                #ifdef USE_HYPERBOLIC_MODEL
                    isf_term_positive_frequency[isf_term_idx] = df*cosh(bo2mt*f); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        isf_term_negative_frequency[isf_term_idx] = df*cosh(bo2mt*f); //FIXME need to calculate new value
                    #endif
                #endif
                #ifdef USE_STANDARD_MODEL
                    isf_term_positive_frequency[isf_term_idx] = df*(exp(-bmt*f) + exp(-t*f)); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        isf_term_negative_frequency[isf_term_idx] = df*(exp(-bmt*f) + exp(-t*f)); //FIXME need to calculate new value
                    #endif
                #endif
                #ifdef USE_NORMALIZATION_MODEL
                    double _num = exp(-bmt*f) + exp(-t*f);
                    double _denom = 1.0 + exp(-beta*f);
                    isf_term_positive_frequency[isf_term_idx] = df*(_num/_denom); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        isf_term_negative_frequency[isf_term_idx] = df*(_num/_denom); //FIXME need to calculate new value
                    #endif
                #endif
            #else
                isf_term_positive_frequency[isf_term_idx] = df*exp(-t*f); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    isf_term_negative_frequency[isf_term_idx] = df*exp(-t*f); //FIXME need to calculate new value
                #endif
            #endif
        }
    }
    
    #ifdef USE_GPU
        //Load isf term onto GPU
        double* d_isf_term_positive_frequency;
        GPU_ASSERT(deac_malloc_device(double, d_isf_term_positive_frequency, genome_size*number_of_timeslices, default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_isf_term_positive_frequency, isf_term_positive_frequency, bytes_isf_term, default_stream));
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            double* d_isf_term_negative_frequency;
            GPU_ASSERT(deac_malloc_device(double, d_isf_term_negative_frequency, genome_size*number_of_timeslices, default_stream));
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
    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        double * population_old_negative_frequency;
        double * population_new_negative_frequency;
        population_old_negative_frequency = (double*) malloc(bytes_population);
        population_new_negative_frequency = (double*) malloc(bytes_population);
    #endif
    for (size_t i=0; i<genome_size*population_size; i++) {
        population_old_positive_frequency[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53; // to_double2
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            population_old_negative_frequency[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53; // to_double2
        #endif
    }
    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        for (size_t i=0; i<population_size; i++) {
            population_new_negative_frequency[i*genome_size] = population_new_positive_frequency[i*genome_size]; // Match up zero (always take value from positive result)
        }
    #endif

    #ifdef USE_GPU
        double* d_population_old_positive_frequency;
        double* d_population_new_positive_frequency;
        GPU_ASSERT(deac_malloc_device(double, d_population_old_positive_frequency, genome_size*population_size, default_stream));
        GPU_ASSERT(deac_malloc_device(double, d_population_new_positive_frequency, genome_size*population_size, default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_population_old_positive_frequency, population_old_positive_frequency, bytes_population, default_stream));
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            double* d_population_old_negative_frequency;
            double* d_population_new_negative_frequency;
            GPU_ASSERT(deac_malloc_device(double, d_population_old_negative_frequency, genome_size*population_size, default_stream));
            GPU_ASSERT(deac_malloc_device(double, d_population_new_negative_frequency, genome_size*population_size, default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_population_old_negative_frequency, population_old_negative_frequency, bytes_population, default_stream));
        #endif
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    // Normalize population
    size_t bytes_normalization = sizeof(double)*population_size;
    size_t bytes_normalization_term = sizeof(double)*genome_size;
    double * normalization;
    double * normalization_term_positive_frequency;
    double * normalization_term_negative_frequency;
    #ifdef USE_GPU
        double* d_normalization;
        double* d_normalization_term_positive_frequency;
        double* d_normalization_term_negative_frequency;
    #endif
    if (normalize) {
        normalization = (double*) malloc(bytes_normalization);
        normalization_term_positive_frequency = (double*) malloc(bytes_normalization_term);
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            normalization_term_negative_frequency = (double*) malloc(bytes_normalization_term);
        #endif
        for (size_t j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            #ifndef ZEROT
                #ifdef USE_HYPERBOLIC_MODEL
                    normalization_term_positive_frequency[j] = df*cosh(0.5*beta*f); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        normalization_term_negative_frequency[j] = df; //FIXME need to calculate new value
                    #endif
                #endif
                #ifdef USE_STANDARD_MODEL
                    normalization_term_positive_frequency[j] = df*(1.0 + exp(-beta*f)); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        normalization_term_negative_frequency[j] = df; //FIXME need to calculate new value
                    #endif
                #endif
                #ifdef USE_NORMALIZATION_MODEL
                    normalization_term_positive_frequency[j] = df; //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        normalization_term_negative_frequency[j] = df; //FIXME need to calculate new value
                    #endif
                #endif
            #endif
            #ifdef ZEROT
                normalization_term_positive_frequency[j] = df; //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    normalization_term_negative_frequency[j] = df; //FIXME need to calculate new value
                #endif
            #endif
        }

        #ifdef USE_GPU
            //Load normalization terms onto GPU
            GPU_ASSERT(deac_malloc_device(double, d_normalization,                         population_size, default_stream));
            GPU_ASSERT(deac_malloc_device(double, d_normalization_term_positive_frequency, genome_size,     default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_normalization_term_positive_frequency, normalization_term_positive_frequency, bytes_normalization_term, default_stream));
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                GPU_ASSERT(deac_malloc_device(double, d_normalization_term_negative_frequency, genome_size,     default_stream));
                GPU_ASSERT(deac_memcpy_host_to_device(d_normalization_term_negative_frequency, normalization_term_negative_frequency, bytes_normalization_term, default_stream));
            #endif
            GPU_ASSERT(deac_wait(default_stream));
        #endif

        //Set normalization
        #ifdef USE_GPU
            //FIXME need gpu_matmul and gpu_normalize_population kernel launchers (and possibly implementations) for CUDA and HIP (also should take a look at SYCL version)
            size_t grid_size_set_normalization = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            GPU_ASSERT(deac_memset(d_normalization, 0, bytes_normalization, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            for (size_t i=0; i<population_size; i++) {
                size_t stream_idx = i % MAX_GPU_STREAMS;
                gpu_matmul(stream_array[stream_idx], d_normalization + i, d_normalization_term_positive_frequency, d_population_old_positive_frequency + genome_size*i, genome_size);
            }
            for (auto& s : stream_array) {
                GPU_ASSERT(deac_wait(s));
            }
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                for (size_t i=0; i<population_size; i++) {
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    gpu_matmul(stream_array[stream_idx], d_normalization + i, d_normalization_term_negative_frequency, d_population_old_negative_frequency + genome_size*i, genome_size);
                }
                for (auto& s : stream_array) {
                    GPU_ASSERT(deac_wait(s));
                }
            #endif

            size_t grid_size_normalize_population = (population_size*genome_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
            gpu_normalize_population(stream_array[0], grid_size_normalize_population, d_population_old_positive_frequency, d_normalization, zeroth_moment, population_size, genome_size); 
            gpu_normalize_population(stream_array[1 % MAX_GPU_STREAMS], grid_size_normalize_population, d_population_old_negative_frequency, d_normalization, zeroth_moment, population_size, genome_size); 
            GPU_ASSERT(deac_wait(stream_array[0]));
            GPU_ASSERT(deac_wait(stream_array[1 % MAX_GPU_STREAMS]));
        #else
            for (size_t i=0; i<population_size; i++) {
                normalization[i] = 0.0;
            }
            matrix_multiply_MxN_by_Nx1(normalization, population_old_positive_frequency,
                    normalization_term_positive_frequency, population_size, genome_size);
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                matrix_multiply_MxN_by_Nx1(normalization, population_old_negative_frequency,
                        normalization_term_negative_frequency, population_size, genome_size);
            #endif
            for (size_t i=0; i<population_size; i++) {
                double _norm = normalization[i];
                for (size_t j=0; j<genome_size; j++) {
                    population_old_positive_frequency[i*genome_size + j] *= zeroth_moment/_norm;
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        population_old_negative_frequency[i*genome_size + j] *= zeroth_moment/_norm;
                    #endif
                }
            }
        #endif
    }

    // Set first moment term
    size_t bytes_first_moments_term = sizeof(double)*genome_size;
    size_t bytes_first_moments = sizeof(double)*population_size;

    double * first_moments;
    double * first_moments_term_positive_frequency;
    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        double * first_moments_term_negative_frequency;
    #endif
    #ifdef USE_GPU
        double* d_first_moments;
        double* d_first_moments_term_positive_frequency;
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            double* d_first_moments_term_negative_frequency;
        #endif
    #endif

    if (use_first_moment) {
        first_moments_term_positive_frequency = (double*) malloc(bytes_first_moments_term);
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            first_moments_term_negative_frequency = (double*) malloc(bytes_first_moments_term);
        #endif
        for (size_t j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            #ifndef ZEROT
                #ifdef USE_HYPERBOLIC_MODEL
                    first_moments_term_positive_frequency[j] = df*f*sinh(0.5*beta*f); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        first_moments_term_negative_frequency[j] = df*f; //FIXME need to calculate new value
                    #endif
                #endif
                #ifdef USE_STANDARD_MODEL
                    first_moments_term_positive_frequency[j] = df*f*(1.0 - exp(-beta*f)); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        first_moments_term_negative_frequency[j] = df*f; //FIXME need to calculate new value
                    #endif
                #endif
                #ifdef USE_NORMALIZATION_MODEL
                    first_moments_term_positive_frequency[j] = df*f*tanh(0.5*beta*f); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        first_moments_term_negative_frequency[j] = df*f; //FIXME need to calculate new value
                    #endif
                #endif
            #else
                first_moments_term_positive_frequency[j] = df*f; //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    first_moments_term_negative_frequency[j] = df*f; //FIXME need to calculate new value
                #endif
            #endif
        }

        first_moments = (double*) malloc(bytes_first_moments);
        #ifdef USE_GPU
            GPU_ASSERT((deac_malloc_device(double, d_first_moments,                         population_size, default_stream));
            GPU_ASSERT((deac_malloc_device(double, d_first_moments_term_positive_frequency, genome_size,     default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_first_moments_term_positive_frequency, first_moments_term_positive_frequency, bytes_first_moments_term, default_stream));
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                GPU_ASSERT((deac_malloc_device(double, d_first_moments_term_negative_frequency, genome_size,     default_stream));
                GPU_ASSERT(deac_memcpy_host_to_device(d_first_moments_term_negative_frequency, first_moments_term_negative_frequency, bytes_first_moments_term, default_stream));
            #endif
            GPU_ASSERT(deac_wait(default_stream));

            size_t grid_size_set_first_moments = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            GPU_ASSERT(deac_memset(d_first_moments, 0, bytes_first_moments, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            for (size_t i=0; i<population_size; i++) {
                size_t stream_idx = i % MAX_GPU_STREAMS;
                gpu_matmul(stream_array[stream_idx], d_first_moments + i, d_first_moments_term_positive_frequency, d_population_old_positive_frequency + genome_size*i, genome_size);
            }
            for (auto& s : stream_array) {
                GPU_ASSERT(deac_wait(s));
            }
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                for (size_t i=0; i<population_size; i++) {
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    gpu_matmul(stream_array[stream_idx], d_first_moments + i, d_first_moments_term_negative_frequency, d_population_old_negative_frequency + genome_size*i, genome_size);
                }
                for (auto& s : stream_array) {
                    GPU_ASSERT(deac_wait(s));
                }
            #endif
        #else
            for (size_t i=0; i<population_size; i++) {
                first_moments[i] = 0.0;
            }
            matrix_multiply_MxN_by_Nx1(first_moments, population_old_positive_frequency,
                    first_moments_term_positive_frequency, population_size, genome_size);
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                matrix_multiply_MxN_by_Nx1(first_moments, population_old_negative_frequency,
                        first_moments_term_negative_frequency, population_size, genome_size);
            #endif
        #endif
    }

    // Set third moment term
    size_t bytes_third_moments_term = sizeof(double)*genome_size;
    size_t bytes_third_moments = sizeof(double)*population_size;

    double * third_moments;
    double * third_moments_term_positive_frequency;
    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        double * third_moments_term_negative_frequency;
    #endif
    #ifdef USE_GPU
        double* d_third_moments;
        double* d_third_moments_term_positive_frequency;
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            double* d_third_moments_term_negative_frequency;
        #endif
    #endif

    if (use_third_moment) {
        third_moments_term_positive_frequency = (double*) malloc(bytes_third_moments_term);
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            third_moments_term_negative_frequency = (double*) malloc(bytes_third_moments_term);
        #endif
        for (size_t j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            #ifndef ZEROT
                #ifdef USE_HYPERBOLIC_MODEL
                    third_moments_term_positive_frequency[j] = df*pow(f,3)*sinh(0.5*beta*f); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        third_moments_term_negative_frequency[j] = df*pow(f,3); //FIXME need to calculate new value
                    #endif
                #endif
                #ifdef USE_STANDARD_MODEL
                    third_moments_term_positive_frequency[j] = df*pow(f,3)*(1.0 - exp(-beta*f)); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        third_moments_term_negative_frequency[j] = df*pow(f,3); //FIXME need to calculate new value
                    #endif
                #endif
                #ifdef USE_NORMALIZATION_MODEL
                    third_moments_term_positive_frequency[j] = df*pow(f,3)*tanh(0.5*beta*f); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        third_moments_term_negative_frequency[j] = df*pow(f,3); //FIXME need to calculate new value
                    #endif
                #endif
            #else
                third_moments_term_positive_frequency[j] = df*pow(f,3); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    third_moments_term_negative_frequency[j] = df*pow(f,3); //FIXME need to calculate new value
                #endif
            #endif
        }

        third_moments = (double*) malloc(bytes_third_moments);
        #ifdef USE_GPU
            GPU_ASSERT((deac_malloc_device(double, d_third_moments,                         population_size, default_stream));
            GPU_ASSERT((deac_malloc_device(double, d_third_moments_term_positive_frequency, genome_size,     default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_third_moments_term_positive_frequency, third_moments_term_positive_frequency, bytes_third_moments_term, default_stream));
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                GPU_ASSERT((deac_malloc_device(double, d_third_moments_term_negative_frequency, genome_size,     default_stream));
                GPU_ASSERT(deac_memcpy_host_to_device(d_third_moments_term_negative_frequency, third_moments_term_negative_frequency, bytes_third_moments_term, default_stream));
            #endif
            GPU_ASSERT(deac_wait(default_stream));

            size_t grid_size_set_third_moments = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            GPU_ASSERT(deac_memset(d_third_moments, 0, bytes_third_moments, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            for (size_t i=0; i<population_size; i++) {
                size_t stream_idx = i % MAX_GPU_STREAMS;
                gpu_matmul(stream_array[stream_idx], d_third_moments + i, d_third_moments_term_positive_frequency, d_population_old_positive_frequency + genome_size*i, genome_size);
            }
            for (auto& s : stream_array) {
                GPU_ASSERT(deac_wait(s));
            }
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                for (size_t i=0; i<population_size; i++) {
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    gpu_matmul(stream_array[stream_idx], d_third_moments + i, d_third_moments_term_negative_frequency, d_population_old_negative_frequency + genome_size*i, genome_size);
                }
                for (auto& s : stream_array) {
                    GPU_ASSERT(deac_wait(s));
                }
            #endif
        #else
            for (size_t i=0; i<population_size; i++) {
                third_moments[i] = 0.0;
            }
            matrix_multiply_MxN_by_Nx1(third_moments, population_old_positive_frequency,
                    third_moments_term_positive_frequency, population_size, genome_size);
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
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
        GPU_ASSERT((deac_malloc_device(double, d_isf_model, number_of_timeslices*population_size, default_stream));
        size_t grid_size_set_isf_model = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
        GPU_ASSERT(deac_memset(d_isf_model, 0, bytes_isf_model, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        for (size_t i=0; i<population_size*number_of_timeslices; i++) {
            size_t stream_idx = i % MAX_GPU_STREAMS;
            size_t _i = i/number_of_timeslices;
            size_t _j = i - _i*number_of_timeslices;
            gpu_matmul(stream_array[stream_idx], d_isf_model + i, d_population_old_positive_frequency + genome_size*_i, d_isf_term_positive_frequency + genome_size*_j, genome_size);
        }
        for (auto& s : stream_array) {
            GPU_ASSERT(deac_wait(s));
        }
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            for (size_t i=0; i<population_size*number_of_timeslices; i++) {
                size_t stream_idx = i % MAX_GPU_STREAMS;
                size_t _i = i/number_of_timeslices;
                size_t _j = i - _i*number_of_timeslices;
                gpu_matmul(stream_array[stream_idx], d_isf_model + i, d_population_old_negative_frequency + genome_size*_i, d_isf_term_negative_frequency + genome_size*_j, genome_size);
            }
            for (auto& s : stream_array) {
                GPU_ASSERT(deac_wait(s));
            }
        #endif
    #else
        for (size_t i=0; i<population_size*number_of_timeslices; i++) {
            isf_model[i] = 0.0;
        }
        matrix_multiply_LxM_by_MxN(isf_model, isf_term_positive_frequency, population_old_positive_frequency,
                number_of_timeslices, genome_size, population_size);
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            matrix_multiply_LxM_by_MxN(isf_model, isf_term_negative_frequency, population_old_negative_frequency,
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
            for (size_t j=0; j<number_of_timeslices; j++) {
                double dt;
                if (j==0) {
                    dt = 0.5*(imaginary_time[j+1] - imaginary_time[j]);
                } else if (j == number_of_timeslices - 1) {
                    dt = 0.5*(imaginary_time[j] - imaginary_time[j-1]);
                } else {
                    dt = 0.5*(imaginary_time[j+1] - imaginary_time[j-1]);
                }
                negative_first_moments_term[j] = dt;
                negative_first_moment += isf[j]*dt;
                negative_first_moment_error = pow(isf_error[j],2) * pow(dt,2);
            }
            negative_first_moment_error = sqrt(negative_first_moment_error);

            negative_first_moments = (double*) malloc(bytes_negative_first_moments);
            #ifdef USE_GPU
                GPU_ASSERT((deac_malloc_device(double, d_negative_first_moments_term, number_of_timeslices, default_stream));
                GPU_ASSERT((deac_malloc_device(double, d_negative_first_moments,      population_size,      default_stream));
                GPU_ASSERT(deac_memcpy_host_to_device(d_negative_first_moments_term, negative_first_moments_term, bytes_negative_first_moments_term, default_stream));
                GPU_ASSERT(deac_wait(default_stream));

                size_t grid_size_set_negative_first_moments = (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                GPU_ASSERT(deac_memset(d_negative_first_moments, 0, bytes_negative_first_moments, default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                for (size_t i=0; i<population_size; i++) {
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    gpu_matmul(stream_array[stream_idx], d_negative_first_moments + i, d_negative_first_moments_term, d_isf_model + number_of_timeslices*i, number_of_timeslices);
                }
                for (auto& s : stream_array) {
                    GPU_ASSERT(deac_wait(s));
                }
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
        GPU_ASSERT((deac_malloc_device(double, d_fitness_old, population_size, default_stream));
        GPU_ASSERT((deac_malloc_device(double, d_fitness_new, population_size, default_stream));

        size_t grid_size_set_fitness = (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
        size_t grid_size_set_fitness_moments = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
        GPU_ASSERT(deac_memset(d_fitness_old, 0, bytes_fitness, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
        for (size_t i=0; i<population_size; i++) {
            size_t stream_idx = i % MAX_GPU_STREAMS;
            gpu_set_fitness(stream_array[stream_idx], d_fitness_old + i, d_isf, d_isf_model + number_of_timeslices*i, d_isf_error, number_of_timeslices);
        }
        for (auto& s : stream_array) {
            GPU_ASSERT(deac_wait(s));
        }
        #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            if (use_negative_first_moment) {
                gpu_set_fitness_moments_reduced_chi_squared(default_stream, grid_size_set_fitness_moments, d_fitness_old, d_negative_first_moments, negative_first_moment, negative_first_moment_error, population_size);
                GPU_ASSERT(deac_wait(default_stream));
            }
        #else
            //FIXME inverse first moment not implemented for single particle fermionic spectral function
        #endif
        if (use_first_moment) {
            gpu_set_fitness_moments_chi_squared(default_stream, grid_size_set_fitness_moments, d_fitness_old, d_first_moments, first_moment, population_size);
            GPU_ASSERT(deac_wait(default_stream));
        }
        if (use_third_moment) {
            gpu_set_fitness_moments_reduced_chi_squared(default_stream, grid_size_set_fitness_moments, d_fitness_old, d_third_moments, third_moment, third_moment_error, population_size);
            GPU_ASSERT(deac_wait(default_stream));
        }
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
                _fitness += pow(first_moments[i] - first_moment,2)/first_moment;
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
    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        double * crossover_probabilities_old_negative_frequency;
        double * crossover_probabilities_new_negative_frequency;
        crossover_probabilities_old_negative_frequency = (double*) malloc(bytes_crossover_probabilities);
        crossover_probabilities_new_negative_frequency = (double*) malloc(bytes_crossover_probabilities);
    #endif

    for (size_t i=0; i<population_size; i++) {
        crossover_probabilities_old_positive_frequency[i] = crossover_probability;
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            crossover_probabilities_old_negative_frequency[i] = crossover_probability;
        #endif
    }

    #ifdef USE_GPU
        double* d_crossover_probabilities_old_positive_frequency;
        double* d_crossover_probabilities_new_positive_frequency;
        GPU_ASSERT((deac_malloc_device(double, d_crossover_probabilities_old_positive_frequency, population_size, default_stream));
        GPU_ASSERT((deac_malloc_device(double, d_crossover_probabilities_new_positive_frequency, population_size, default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_crossover_probabilities_old_positive_frequency, crossover_probabilities_old_positive_frequency, bytes_crossover_probabilities, default_stream));
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            double* d_crossover_probabilities_old_negative_frequency;
            double* d_crossover_probabilities_new_negative_frequency;
            GPU_ASSERT((deac_malloc_device(double, d_crossover_probabilities_old_negative_frequency, population_size, default_stream));
            GPU_ASSERT((deac_malloc_device(double, d_crossover_probabilities_new_negative_frequency, population_size, default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_crossover_probabilities_old_negative_frequency, crossover_probabilities_old_negative_frequency, bytes_crossover_probabilities, default_stream));
        #endif
        GPU_ASSERT(deac_wait(default_stream));
    #endif

    size_t bytes_differential_weights = sizeof(double)*population_size;
    double * differential_weights_old_positive_frequency;
    double * differential_weights_new_positive_frequency;
    differential_weights_old_positive_frequency = (double*) malloc(bytes_differential_weights);
    differential_weights_new_positive_frequency = (double*) malloc(bytes_differential_weights);
    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        double * differential_weights_old_negative_frequency;
        double * differential_weights_new_negative_frequency;
        differential_weights_old_negative_frequency = (double*) malloc(bytes_differential_weights);
        differential_weights_new_negative_frequency = (double*) malloc(bytes_differential_weights);
    #endif

    for (size_t i=0; i<population_size; i++) {
        differential_weights_old_positive_frequency[i] = differential_weight;
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            differential_weights_old_negative_frequency[i] = differential_weight;
        #endif
    }

    #ifdef USE_GPU
        double* d_differential_weights_old_positive_frequency;
        double* d_differential_weights_new_positive_frequency;
        GPU_ASSERT((deac_malloc_device(double, d_differential_weights_old_positive_frequency, population_size, default_stream));
        GPU_ASSERT((deac_malloc_device(double, d_differential_weights_new_positive_frequency, population_size, default_stream));
        GPU_ASSERT(deac_memcpy_host_to_device(d_differential_weights_old_positive_frequency, differential_weights_old_positive_frequency, bytes_differential_weights, default_stream));
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            double* d_differential_weights_old_negative_frequency;
            double* d_differential_weights_new_negative_frequency;
            GPU_ASSERT((deac_malloc_device(double, d_differential_weights_old_negative_frequency, population_size, default_stream));
            GPU_ASSERT((deac_malloc_device(double, d_differential_weights_new_negative_frequency, population_size, default_stream));
            GPU_ASSERT(deac_memcpy_host_to_device(d_differential_weights_old_negative_frequency, differential_weights_old_negative_frequency, bytes_differential_weights, default_stream));
        #endif
        q.wait();
    #endif

    //Initialize statistics arrays
    double* fitness_mean;
    double* fitness_minimum;
    double* fitness_squared_mean;
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
            size_t grid_size_set_stats = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            GPU_ASSERT((deac_malloc_device(double, d_fitness_mean        , number_of_generations, default_stream));
            GPU_ASSERT((deac_malloc_device(double, d_fitness_squared_mean, number_of_generations, default_stream));
            GPU_ASSERT(deac_memset(d_fitness_mean,         0, bytes_fitness_mean,         default_stream));
            GPU_ASSERT(deac_memset(d_fitness_squared_mean, 0, bytes_fitness_squared_mean, default_stream));
        #endif
    }
    
    size_t bytes_mutate_indices = sizeof(bool)*genome_size*population_size;
    bool* mutate_indices_positive_frequency;
    mutate_indices_positive_frequency = (bool*) malloc(bytes_mutate_indices);
    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        bool* mutate_indices_negative_frequency;
        mutate_indices_negative_frequency = (bool*) malloc(bytes_mutate_indices);
    #endif
    #ifdef USE_GPU
        bool* d_mutate_indices_positive_frequency;
        GPU_ASSERT((deac_malloc_device(bool, d_mutate_indices_positive_frequency, population_size*genome_size, default_stream));
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            bool* d_mutate_indices_negative_frequency;
            GPU_ASSERT((deac_malloc_device(bool, d_mutate_indices_negative_frequency, population_size*genome_size, default_stream));
        #endif

        size_t bytes_rejection_indices = sizeof(bool)*population_size;
        bool* d_rejection_indices;
        GPU_ASSERT((deac_malloc_device(bool, d_rejection_indices, population_size, default_stream));
    #endif

    size_t* mutant_indices;
    size_t bytes_mutant_indices = sizeof(size_t)*3*population_size;
    mutant_indices = (size_t*) malloc(bytes_mutant_indices);
    #ifdef USE_GPU
        size_t* d_mutant_indices;
        GPU_ASSERT((deac_malloc_device(size_t, d_mutant_indices, 3*population_size, default_stream));
    #endif

    double minimum_fitness;
    size_t minimum_fitness_idx;
    #ifdef USE_GPU
        size_t bytes_minimum_fitness = sizeof(double);
        double* d_minimum_fitness;
        GPU_ASSERT((deac_malloc_device(bool, d_minimum_fitness, 1, default_stream));
    #endif

    #ifdef USE_GPU
        size_t bytes_rng_state = sizeof(uint64_t)*4*population_size*(genome_size + 1); //FIXME FIXME FIXME might need bigger state for negative frequency space (double check this calculation)

        // Generate rng state
        uint64_t* d_rng_state;
        uint64_t* rng_state;
        rng_state = (uint64_t *) malloc(bytes_rng_state);

        for (size_t i=0; i<population_size; i++) {
            for (size_t j=0; j < genome_size + 1; j++) {
                xoshiro256p_copy_state(rng_state + 4*(i*genome_size + j), rng->s);
                xoshiro256p_jump(rng->s);
            }
        }
        GPU_ASSERT((deac_malloc_device(uint64_t, d_rng_state, 4*population_size*(genome_size + 1), default_stream)); //FIXME FIXME FIXME might need bigger state for negative frequency (double check this calculation)
        GPU_ASSERT(deac_memcpy_host_to_device(d_rng_state, rng_state, bytes_rng_state, default_stream));
        GPU_ASSERT(deac_wait(default_stream));
    #endif
    
    size_t generation;
    for (size_t ii=0; ii < number_of_generations - 1; ii++) {
        generation = ii;
        #ifdef USE_GPU
            gpu_get_minimum_fitness(default_stream, d_fitness_old, d_minimum_fitness, population_size);
            GPU_ASSERT(deac_memcpy_device_to_host(&minimum_fitness, d_minimum_fitness, bytes_minimum_fitness, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
        #else
            minimum_fitness = minimum(fitness_old, population_size);
        #endif

        //Get Statistics
        if (track_stats) {
            #ifdef USE_GPU
                size_t grid_size_set_stats = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
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
            gpu_set_crossover_probabilities_new(default_stream, grid_size_self_adapting_parameters, d_rng_state, d_crossover_probabilities_new_positive_frequency, d_crossover_probabilities_old_positive_frequency, self_adapting_crossover_probability, population_size);
            gpu_set_differential_weights_new(default_stream, grid_size_self_adapting_parameters, d_rng_state + 4*population_size, d_differential_weights_new_positive_frequency, d_differential_weights_old_positive_frequency, self_adapting_differential_weight_probability, population_size);
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                gpu_set_crossover_probabilities_new(default_stream, grid_size_self_adapting_parameters, d_rng_state, d_crossover_probabilities_new_negative_frequency, d_crossover_probabilities_old_negative_frequency, self_adapting_crossover_probability, population_size);
                gpu_set_differential_weights_new(default_stream, grid_size_self_adapting_parameters, d_rng_state + 4*population_size, d_differential_weights_new_negative_frequency, d_differential_weights_old_negative_frequency, self_adapting_differential_weight_probability, population_size);
            #endif
            GPU_ASSERT(deac_wait(default_stream));
        #else
            //Set crossover probabilities and differential weights
            for (size_t i=0; i<population_size; i++) {
                if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
                    crossover_probabilities_new_positive_frequency[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53;
                } else {
                    crossover_probabilities_new_positive_frequency[i] = crossover_probabilities_old_positive_frequency[i];
                }

                if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
                    //differential_weights_new[i] = 
                    //    self_adapting_differential_weight_shift + 
                    //    self_adapting_differential_weight*((xoshiro256p(rng) >> 11) * 0x1.0p-53);
                    differential_weights_new_positive_frequency[i] = 2.0*((xoshiro256p(rng) >> 11) * 0x1.0p-53);
                } else {
                    differential_weights_new_positive_frequency[i] = differential_weights_old_positive_frequency[i];
                }

                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
                        crossover_probabilities_new_negative_frequency[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53;
                    } else {
                        crossover_probabilities_new_negative_frequency[i] = crossover_probabilities_old_negative_frequency[i];
                    }

                    if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
                        //differential_weights_new[i] = 
                        //    self_adapting_differential_weight_shift + 
                        //    self_adapting_differential_weight*((xoshiro256p(rng) >> 11) * 0x1.0p-53);
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
            gpu_set_mutant_indices(default_stream, grid_size_set_mutant_indices, d_rng_state, d_mutant_indices, population_size);
            gpu_set_mutate_indices(default_stream, grid_size_set_mutate_indices, d_rng_state + 4*population_size, d_mutate_indices_positive_frequency, d_crossover_probabilities_new_positive_frequency, population_size, genome_size);
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                gpu_set_mutate_indices(default_stream, grid_size_set_mutate_indices, d_rng_state + 4*population_size, d_mutate_indices_negative_frequency, d_crossover_probabilities_new_negative_frequency, population_size, genome_size);
            #endif
            GPU_ASSERT(deac_wait(default_stream));
        #else
            //Set mutant population and indices 
            for (size_t i=0; i<population_size; i++) {
                set_mutant_indices(rng, mutant_indices + 3*i, i, population_size);
                double crossover_rate_positive_frequency = crossover_probabilities_new_positive_frequency[i];
                for (size_t j=0; j<genome_size; j++) {
                    mutate_indices_positive_frequency[i*genome_size + j] = (xoshiro256p(rng) >> 11) * 0x1.0p-53 < crossover_rate_positive_frequency;
                }
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    double crossover_rate_negative_frequency = crossover_probabilities_new_negative_frequency[i];
                    for (size_t j=0; j<genome_size; j++) {
                        mutate_indices_negative_frequency[i*genome_size + j] = (xoshiro256p(rng) >> 11) * 0x1.0p-53 < crossover_rate_negative_frequency;
                    }
                #endif
            }
        #endif

        #ifdef USE_GPU
            size_t grid_size_set_population_new = (population_size*genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            gpu_set_population_new(default_stream, grid_size_set_population_new, d_population_new_positive_frequency, d_population_old_positive_frequency, d_mutant_indices, d_differential_weights_new_positive_frequency, d_mutate_indices_positive_frequency, population_size, genome_size);
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                gpu_set_population_new(default_stream, grid_size_set_population_new, d_population_new_negative_frequency, d_population_old_negative_frequency, d_mutant_indices, d_differential_weights_new_negative_frequency, d_mutate_indices_negative_frequency, population_size, genome_size);
            #endif
            GPU_ASSERT(deac_wait(default_stream));

            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                size_t grid_size_match_population_zero = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                gpu_match_population_zero(default_stream, grid_size_match_population_zero, d_population_new_negative_frequency, d_population_new_positive_frequency, population_size, genome_size);
                GPU_ASSERT(deac_wait(default_stream));
            #endif
        #else
            for (size_t i=0; i<population_size; i++) {
                double F_positive_frequency = differential_weights_new_positive_frequency[i];
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    double F_negative_frequency = differential_weights_new_negative_frequency[i];
                #endif
                size_t mutant_index1 = mutant_indices[3*i];
                size_t mutant_index2 = mutant_indices[3*i + 1];
                size_t mutant_index3 = mutant_indices[3*i + 2];
                for (size_t j=0; j<genome_size; j++) {
                    bool mutate_positive_frequency = mutate_indices_positive_frequency[i*genome_size + j];
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        bool mutate_negative_frequency = mutate_indices_negative_frequency[i*genome_size + j];
                    #endif
                    if (mutate_positive_frequency) {
                        population_new_positive_frequency[i*genome_size + j] = fabs( 
                            population_old_positive_frequency[mutant_index1*genome_size + j] + F_positive_frequency*(
                                    population_old_positive_frequency[mutant_index2*genome_size + j] -
                                    population_old_positive_frequency[mutant_index3*genome_size + j]));
                    } else {
                        population_new_positive_frequency[i*genome_size + j] = population_old_positive_frequency[i*genome_size + j];
                    }
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        if (mutate_negative_frequency) {
                            population_new_negative_frequency[i*genome_size + j] = fabs( 
                                population_old_negative_frequency[mutant_index1*genome_size + j] + F_negative_frequency*(
                                        population_old_negative_frequency[mutant_index2*genome_size + j] -
                                        population_old_negative_frequency[mutant_index3*genome_size + j]));
                        } else {
                            population_new_negative_frequency[i*genome_size + j] = population_old_negative_frequency[i*genome_size + j];
                        }
                        if (j == 0) {
                            // Set zero frequency to same value
                            population_new_negative_frequency[i*genome_size + j] = population_new_positive_frequency[i*genome_size + j];
                        }
                    #endif
                }
            }
        #endif

        // Normalization
        if (normalize) {
            #ifdef USE_GPU
                size_t grid_size_set_normalization = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                GPU_ASSERT(deac_memset(d_normalization, 0, bytes_normalization, default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                for (size_t i=0; i<population_size; i++) {
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    gpu_matmul(stream_array[stream_idx], d_normalization + i, d_normalization_term_positive_frequency, d_population_new_positive_frequency + genome_size*i, genome_size);
                }
                for (auto& s : stream_array) {
                    GPU_ASSERT(deac_wait(s));
                }
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    for (size_t i=0; i<population_size; i++) {
                        size_t stream_idx = i % MAX_GPU_STREAMS;
                        gpu_matmul(stream_array[stream_idx], d_normalization + i, d_normalization_term_negative_frequency, d_population_new_negative_frequency + genome_size*i, genome_size);
                    }
                    for (auto& s : stream_array) {
                        GPU_ASSERT(deac_wait(s));
                    }
                #endif

                size_t grid_size_normalize_population = (population_size*genome_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
                gpu_normalize_population(default_stream, grid_size_normalize_population, d_population_new_positive_frequency, d_normalization, zeroth_moment, population_size, genome_size); 
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    gpu_normalize_population(default_stream, grid_size_normalize_population, d_population_new_negative_frequency, d_normalization, zeroth_moment, population_size, genome_size); 
                #endif
                GPU_ASSERT(deac_wait(default_stream));
            #else
                for (size_t i=0; i<population_size; i++) {
                    normalization[i] = 0.0;
                }
                #ifndef SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION
                #else
                    matrix_multiply_MxN_by_Nx1(normalization, population_new_positive_frequency,
                            normalization_term_positive_frequency, population_size, genome_size);
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        matrix_multiply_MxN_by_Nx1(normalization, population_new_negative_frequency,
                                normalization_term_negative_frequency, population_size, genome_size);
                    #endif
                    for (size_t i=0; i<population_size; i++) {
                        double _norm = normalization[i];
                        for (size_t j=0; j<genome_size; j++) {
                            population_new_positive_frequency[i*genome_size + j] *= zeroth_moment/_norm;
                            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                                population_new_negative_frequency[i*genome_size + j] *= zeroth_moment/_norm;
                            #endif
                        }
                    }
                #endif
            #endif
        }

        //Rejection
        //Set model isf for new population
        #ifdef USE_GPU
            size_t grid_size_set_isf_model = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            GPU_ASSERT(deac_memset(d_isf_model, 0, bytes_isf_model, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            for (size_t i=0; i<population_size*number_of_timeslices; i++) {
                size_t stream_idx = i % MAX_GPU_STREAMS;
                size_t _i = i/number_of_timeslices;
                size_t _j = i - _i*number_of_timeslices;
                gpu_matmul(stream_array[stream_idx], d_isf_model + i, d_population_new_positive_frequency + genome_size*_i, d_isf_term_positive_frequency + genome_size*_j, genome_size);
            }
            for (auto& s : stream_array) {
                GPU_ASSERT(deac_wait(s));
            }
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                for (size_t i=0; i<population_size*number_of_timeslices; i++) {
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    size_t _i = i/number_of_timeslices;
                    size_t _j = i - _i*number_of_timeslices;
                    gpu_matmul(stream_array[stream_idx], d_isf_model + i, d_population_new_negative_frequency + genome_size*_i, d_isf_term_negative_frequency + genome_size*_j, genome_size);
                }
                for (auto& s : stream_array) {
                    GPU_ASSERT(deac_wait(s));
                }
            #endif
        #else
            for (size_t i=0; i<population_size*number_of_timeslices; i++) {
                isf_model[i] = 0.0;
            }
            matrix_multiply_LxM_by_MxN(isf_model, isf_term_positive_frequency, population_new_positive_frequency,
                    number_of_timeslices, genome_size, population_size);
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                matrix_multiply_LxM_by_MxN(isf_model, isf_term_negative_frequency, population_new_negative_frequency,
                        number_of_timeslices, genome_size, population_size);
            #endif
        #endif

        //Set moments
        if (use_negative_first_moment) {
            #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                #ifdef USE_GPU
                    size_t grid_size_set_negative_first_moments = (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                    GPU_ASSERT(deac_memset(d_negative_first_moments, 0, bytes_negative_first_moments, default_stream));
                    GPU_ASSERT(deac_wait(default_stream));
                    for (size_t i=0; i<population_size; i++) {
                        size_t stream_idx = i % MAX_GPU_STREAMS;
                        gpu_matmul(stream_array[stream_idx], d_negative_first_moments + i, d_negative_first_moments_term, d_isf_model + number_of_timeslices*i, number_of_timeslices);
                    }
                    for (auto& s : stream_array) {
                        GPU_ASSERT(deac_wait(s));
                    }
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
                size_t grid_size_set_first_moments = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                GPU_ASSERT(deac_memset(d_first_moments, 0, bytes_first_moments, default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                for (size_t i=0; i<population_size; i++) {
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    gpu_matmul(stream_array[stream_idx], d_first_moments + i, d_first_moments_term_positive_frequency, d_population_new_positive_frequency + genome_size*i, genome_size);
                }
                for (auto& s : stream_array) {
                    GPU_ASSERT(deac_wait(s));
                }
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    for (size_t i=0; i<population_size; i++) {
                        size_t stream_idx = i % MAX_GPU_STREAMS;
                        gpu_matmul(stream_array[stream_idx], d_first_moments + i, d_first_moments_term_negative_frequency, d_population_new_negative_frequency + genome_size*i, genome_size);
                    }
                    for (auto& s : stream_array) {
                        GPU_ASSERT(deac_wait(s));
                    }
                #endif
            #else
                for (size_t i=0; i<population_size; i++) {
                    first_moments[i] = 0.0;
                }
                matrix_multiply_MxN_by_Nx1(first_moments, population_new_positive_frequency,
                        first_moments_term_positive_frequency, population_size, genome_size);
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    matrix_multiply_MxN_by_Nx1(first_moments, population_new_negative_frequency,
                            first_moments_term_negative_frequency, population_size, genome_size);
                #endif
            #endif
        }
        if (use_third_moment) {
            #ifdef USE_GPU
                size_t grid_size_set_third_moments = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                GPU_ASSERT(deac_memset(d_third_moments, 0, bytes_third_moments, default_stream));
                GPU_ASSERT(deac_wait(default_stream));
                for (size_t i=0; i<population_size; i++) {
                    size_t stream_idx = i % MAX_GPU_STREAMS;
                    gpu_matmul(stream_array[stream_idx], d_third_moments + i, d_third_moments_term_positive_frequency, d_population_new_positive_frequency + genome_size*i, genome_size);
                }
                for (auto& s : stream_array) {
                    GPU_ASSERT(deac_wait(s));
                }
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    for (size_t i=0; i<population_size; i++) {
                        size_t stream_idx = i % MAX_GPU_STREAMS;
                        gpu_matmul(stream_array[stream_idx], d_third_moments + i, d_third_moments_term_negative_frequency, d_population_new_negative_frequency + genome_size*i, genome_size);
                    }
                    for (auto& s : stream_array) {
                        GPU_ASSERT(deac_wait(s));
                    }
                #endif
            #else
                for (size_t i=0; i<population_size; i++) {
                    third_moments[i] = 0.0;
                }
                matrix_multiply_MxN_by_Nx1(third_moments, population_new_positive_frequency,
                        third_moments_term_positive_frequency, population_size, genome_size);
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    matrix_multiply_MxN_by_Nx1(third_moments, population_new_negative_frequency,
                            third_moments_term_negative_frequency, population_size, genome_size);
                #endif
            #endif
        }

        //Set fitness for new population
        #ifdef USE_GPU
            size_t grid_size_set_fitness = (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            size_t grid_size_set_fitness_moments = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            GPU_ASSERT(deac_memset(d_fitness_new, 0, bytes_fitness, default_stream));
            GPU_ASSERT(deac_wait(default_stream));
            for (size_t i=0; i<population_size; i++) {
                size_t stream_idx = i % MAX_GPU_STREAMS;
                gpu_set_fitness(stream_array[stream_idx], d_fitness_new + i, d_isf, d_isf_model + number_of_timeslices*i, d_isf_error, number_of_timeslices);
            }
            for (auto& s : stream_array) {
                GPU_ASSERT(deac_wait(s));
            }
            #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                if (use_negative_first_moment) {
                    gpu_set_fitness_moments_reduced_chi_squared(default_stream, grid_size_set_fitness_moments, d_fitness_new, d_negative_first_moments, negative_first_moment, negative_first_moment_error, population_size);
                    GPU_ASSERT(deac_wait(default_stream));
                }
            #else
                //FIXME inverse first moment not implemented
            #endif
            if (use_first_moment) {
                gpu_set_fitness_moments_chi_squared(default_stream, grid_size_set_fitness_moments, d_fitness_new, d_first_moments, first_moment, population_size);
                GPU_ASSERT(deac_wait(default_stream));
            }
            if (use_third_moment) {
                gpu_set_fitness_moments_reduced_chi_squared(default_stream, grid_size_set_fitness_moments, d_fitness_new, d_third_moments, third_moment, third_moment_error, population_size);
                GPU_ASSERT(deac_wait(default_stream));
            }
        #else
            // Fitness set in rejection step
        #endif

        //Rejection step
        #ifdef USE_GPU
            size_t grid_size_set_rejection_indices = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            size_t grid_size_swap_control_parameters = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            size_t grid_size_swap_populations = (population_size*genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;

            gpu_set_rejection_indices(default_stream, grid_size_set_rejection_indices, d_rejection_indices, d_fitness_new, d_fitness_old, population_size);
            GPU_ASSERT(deac_wait(default_stream));

            gpu_swap_control_parameters(default_stream, grid_size_swap_control_parameters, d_crossover_probabilities_old_positive_frequency, d_crossover_probabilities_new_positive_frequency, d_rejection_indices, population_size);
            gpu_swap_control_parameters(default_stream, grid_size_swap_control_parameters, d_differential_weights_old_positive_frequency, d_differential_weights_new_positive_frequency, d_rejection_indices, population_size);
            gpu_swap_populations(default_stream, grid_size_swap_populations, d_population_old_positive_frequency, d_population_new_positive_frequency, d_rejection_indices, population_size, genome_size);

            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                gpu_swap_control_parameters(default_stream, grid_size_swap_control_parameters, d_crossover_probabilities_old_negative_frequency, d_crossover_probabilities_new_negative_frequency, d_rejection_indices, population_size);
                gpu_swap_control_parameters(default_stream, grid_size_swap_control_parameters, d_differential_weights_old_negative_frequency, d_differential_weights_new_negative_frequency, d_rejection_indices, population_size);
                gpu_swap_populations(default_stream, grid_size_swap_populations, d_population_old_negative_frequency, d_population_new_negative_frequency, d_rejection_indices, population_size, genome_size);
            #endif
            GPU_ASSERT(deac_wait(default_stream));
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
                    //FIXME inverse first moment not implemented for single particle fermionic spectral function
                #endif
                if (use_first_moment) {
                    _fitness += pow(first_moments[i] - first_moment,2)/first_moment;
                }
                if (use_third_moment) {
                    _fitness += pow((third_moment - third_moments[i])/third_moment_error,2);
                }
                // Rejection step
                if (_fitness <= fitness_old[i]) {
                    fitness_old[i] = _fitness;
                    crossover_probabilities_old_positive_frequency[i] = crossover_probabilities_new_positive_frequency[i];
                    differential_weights_old_positive_frequency[i] = differential_weights_new_positive_frequency[i];
                    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                        crossover_probabilities_old_negative_frequency[i] = crossover_probabilities_new_negative_frequency[i];
                        differential_weights_old_negative_frequency[i] = differential_weights_new_negative_frequency[i];
                    #endif
                    for (size_t j=0; j<genome_size; j++) {
                        population_old_positive_frequency[i*genome_size + j] = population_new_positive_frequency[i*genome_size + j];
                        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                            population_old_negative_frequency[i*genome_size + j] = population_new_negative_frequency[i*genome_size + j];
                        #endif
                    }
                    #endif
                }
            }
        #endif
    }

    //Transfer data from gpu to host
    #ifdef USE_GPU
        GPU_ASSERT(deac_memcpy_device_to_host(fitness_old, d_fitness_old, bytes_fitness, stream_array[0]));
        GPU_ASSERT(deac_memcpy_device_to_host(population_old_positive_frequency, d_population_old_positive_frequency, bytes_population, stream_array[1 % MAX_GPU_STREAMS]));
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
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
    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        best_dsf = (double*) malloc(sizeof(double)*(2*genome_size - 1));
        best_frequency = (double*) malloc(sizeof(double)*(2*genome_size - 1));
    #else
        best_dsf = (double*) malloc(sizeof(double)*genome_size);
        best_frequency = (double*) malloc(sizeof(double)*genome_size);
    #endif
    for (size_t i=0; i<genome_size; i++) {
        double f = frequency[i];
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            size_t idx_p = genome_size + i - 1;
            size_t idx_n = genome_size - i - 1;
        #else
            size_t idx_p = i;
        #endif
        best_frequency[idx_p] = f;
        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
            best_frequency[idx_n] = -f;
        #endif
        #ifndef ZEROT
            #ifdef USE_HYPERBOLIC_MODEL
                best_dsf[idx_p] = 0.5*population_old_positive_frequency[genome_size*minimum_fitness_idx + i]*exp(0.5*beta*f); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                     best_dsf[idx_n] = population_old_negative_frequency[genome_size*minimum_fitness_idx + i]; //FIXME need to calculate new value
                #endif
            #endif
            #ifdef USE_STANDARD_MODEL
                best_dsf[idx_p] = population_old_positive_frequency[genome_size*minimum_fitness_idx + i]; //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    best_dsf[idx_n] = population_old_negative_frequency[genome_size*minimum_fitness_idx + i]; //FIXME need to calculate new value
                #endif
            #endif
            #ifdef USE_NORMALIZATION_MODEL
                best_dsf[idx_p] = population_old_positive_frequency[genome_size*minimum_fitness_idx + i]/(1.0 + exp(-beta*f)); //FIXME this might be wrong when not using bosonic detailed balance condition for isf
                #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                    best_dsf[idx_n] = population_old_negative_frequency[genome_size*minimum_fitness_idx + i]; //FIXME need to calculate new value
                #endif
            #endif
        #endif
        #ifdef ZEROT
            best_dsf[idx_p] = population_old_positive_frequency[genome_size*minimum_fitness_idx + i]; //FIXME this might be wrong when not using bosonic detailed balance condition for isf
            #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
                best_dsf[idx_n] = population_old_negative_frequency[genome_size*minimum_fitness_idx + i]; //FIXME need to calculate new value
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

    //Save data
    #ifndef ZEROT
        std::string deac_prefix = "deac-" + spectra_type;
    #else
        std::string deac_prefix = "deac-zT" + spectra_type;
    #endif
    std::string best_dsf_filename_str = string_format("%s_dsf_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
    fs::path best_dsf_filename = save_directory / best_dsf_filename_str;
    std::string frequency_filename_str = string_format("%s_frequency_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
    fs::path frequency_filename = save_directory / frequency_filename_str;
    #ifdef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
        write_array(best_dsf_filename, best_dsf, genome_size);
        write_array(frequency_filename, best_frequency, genome_size);
    #else
        write_array(best_dsf_filename, best_dsf, 2*genome_size - 1);
        write_array(frequency_filename, best_frequency, 2*genome_size - 1);
    #endif
    fs::path fitness_mean_filename;
    fs::path fitness_minimum_filename;
    fs::path fitness_squared_mean_filename;
    if (track_stats) {
        std::string fitness_mean_filename_str = string_format("%s_stats_fitness-mean_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
        std::string fitness_minimum_filename_str = string_format("%s_stats_fitness-minimum_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
        std::string fitness_squared_mean_filename_str = string_format("%s_stats_fitness-squared-mean_%s.bin",deac_prefix.c_str(),uuid_str.c_str());
        fs::path fitness_mean_filename = save_directory / fitness_mean_filename_str;
        fs::path fitness_minimum_filename = save_directory / fitness_minimum_filename_str;
        fs::path fitness_squared_mean_filename = save_directory / fitness_squared_mean_filename_str;
        write_array(fitness_mean_filename, fitness_mean, generation + 1);
        write_array(fitness_minimum_filename, fitness_minimum, generation + 1);
        write_array(fitness_squared_mean_filename, fitness_squared_mean, generation + 1);
    }

    //Write to log file
    std::string log_filename_str = string_format("%s_log_%s.dat",deac_prefix.c_str(),uuid_str.c_str());
    fs::path log_filename = save_directory / log_filename_str;
    std::ofstream log_ofs(log_filename.c_str(), std::ios_base::out | std::ios_base::app );

    //Build Type
    #ifdef USE_HYPERBOLIC_MODEL
        log_ofs << "build: USE_HYPERBOLIC_MODEL" << std::endl;
    #endif
    #ifdef USE_STANDARD_MODEL
        log_ofs << "build: USE_STANDARD_MODEL" << std::endl;
    #endif
    #ifdef USE_NORMALIZATION_MODEL
        log_ofs << "build: USE_NORMALIZATION_MODEL" << std::endl;
    #endif
    log_ofs << "kernel: " << spectra_type << std::endl;

    //Input parameters
    log_ofs << "temperature: " << temperature << std::endl;
    log_ofs << "number_of_generations: " << number_of_generations << std::endl;
    log_ofs << "number_of_timeslices: " << number_of_timeslices << std::endl;
    log_ofs << "population_size: " << population_size << std::endl;
    log_ofs << "genome_size: " << genome_size << std::endl;
    log_ofs << "normalize: " << normalize << std::endl;
    log_ofs << "use_negative_first_moment: " << use_negative_first_moment << std::endl;
    log_ofs << "first_moment: " << first_moment << std::endl;
    log_ofs << "third_moment: " << third_moment << std::endl;
    log_ofs << "third_moment_error: " << third_moment_error << std::endl;
    log_ofs << "crossover_probability: " << crossover_probability << std::endl;
    log_ofs << "self_adapting_crossover_probability: " << self_adapting_crossover_probability << std::endl;
    log_ofs << "differential_weight: " << differential_weight << std::endl;
    log_ofs << "self_adapting_differential_weight_probability: " << self_adapting_differential_weight_probability << std::endl;
    log_ofs << "self_adapting_differential_weight_shift: " << self_adapting_differential_weight_shift << std::endl;
    log_ofs << "self_adapting_differential_weight: " << self_adapting_differential_weight << std::endl;
    log_ofs << "stop_minimum_fitness: " << stop_minimum_fitness << std::endl;
    log_ofs << "track_stats: " << track_stats << std::endl;
    log_ofs << "seed: " << seed << std::endl;

    //Generated variables
    log_ofs << "best_dsf_filename: " << best_dsf_filename << std::endl;
    log_ofs << "frequency_filename: " << frequency_filename << std::endl;
    log_ofs << "generation: " << generation << std::endl;
    std::cout << "generation: " << generation << std::endl;
    log_ofs << "minimum_fitness: " << minimum_fitness << std::endl;
    std::cout << "minimum_fitness: " << minimum_fitness << std::endl;
    if (track_stats) {
        log_ofs << "fitness_mean_filename: " << fitness_mean_filename << std::endl;
        log_ofs << "fitness_minimum_filename: " << fitness_minimum_filename << std::endl;
        log_ofs << "fitness_squared_mean_filename: " << fitness_squared_mean_filename << std::endl;
    }
    log_ofs.close();

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
        //FIXME need to add inverse first moment functionality for negative frequency, then can remove this ifdef
        if (use_negative_first_moment) {
            free(negative_first_moments_term_positive_frequency);
            free(negative_first_moments);
        }
    #endif
    free(crossover_probabilities_old_positive_frequency);
    free(crossover_probabilities_new_positive_frequency);
    free(differential_weights_old_positive_frequency);
    free(differential_weights_new_positive_frequency);
    free(mutate_indices_positive_frequency);

    #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
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
            GPU_ASSERT(deac_free(d_normalization_term_positive_frequency, stream_array[15 % MAX_GPU_STREAMS]));
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
            //FIXME need to add inverse first moment functionality for negative frequency, then can remove this ifdef
            if (use_negative_first_moment) {
                GPU_ASSERT(deac_free(d_negative_first_moments,                         stream_array[20 % MAX_GPU_STREAMS]));
                GPU_ASSERT(deac_free(d_negative_first_moments_term_positive_frequency, stream_array[21 % MAX_GPU_STREAMS]));
            }
        #endif
        GPU_ASSERT(deac_free(d_crossover_probabilities_old_positive_frequency, stream_array[22 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_crossover_probabilities_new_positive_frequency, stream_array[23 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_differential_weights_old_positive_frequency,    stream_array[24 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_differential_weights_new_positive_frequency,    stream_array[25 % MAX_GPU_STREAMS]));
        GPU_ASSERT(deac_free(d_mutate_indices_positive_frequency,              stream_array[26 % MAX_GPU_STREAMS]));

        #ifndef USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
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
    #endif
}

int main (int argc, char *argv[]) {
    argparse::ArgumentParser program("deac-cpp", "2.0.0-rc1");
    program.add_argument("-T", "--temperature")
        .help("Temperature of system.")
        .default_value(0.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-N", "--number_of_generations")
        .help("Number of generations before genetic algorithm quits.")
        .default_value(100000)
        .action([](const std::string& value) { return std::stoul(value); });
    program.add_argument("-P","--population_size")
        .help("Size of initial population")
        .default_value(512)
        .action([](const std::string& value) { return std::stoul(value); });
    program.add_argument("-M","--genome_size")
        .help("Size of genome.")
        .default_value(512)
        .action([](const std::string& value) { return std::stoul(value); });
    program.add_argument("--omega_max")
        .help("Maximum frequency to explore.")
        .default_value(60.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--frequency_file")
        .help("Filename containing frequency partition (genome_size and omega_max will be ignored).");
    program.add_argument("--normalize")
        .help("Normalize spectrum to the zeroeth moment.")
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
    program.add_argument("--third_moment")
        .help("Set third frequency moment and use in fitness function.")
        .default_value(-1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--third_moment_error")
        .help("Set error for third frequency moment.")
        .default_value(0.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-r","--crossover_probability")
        .help("Initial probability for parent gene to become mutant vector gene.")
        .default_value(0.9)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-u","--self_adapting_crossover_probability")
        .help("Probability for `crossover_probability` to mutate.")
        .default_value(0.1)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-F","--differential_weight")
        .help("Initial weight factor when creating mutant vector.")
        .default_value(0.9)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-v","--self_adapting_differential_weight_probability")
        .help("Probability for `differential_weight` to mutate.")
        .default_value(0.1)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-l","--self_adapting_differential_weight_shift")
        .help("If `self_adapting_differential_weight_probability` mutate, new value is `l + m*rand()`.")
        .default_value(0.1)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-m","--self_adapting_differential_weight")
        .help("If `self_adapting_differential_weight_probability` mutate, new value is `l + m*rand()`.")
        .default_value(0.9)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--stop_minimum_fitness")
        .help("Stopping criteria, if minimum fitness is below `stop_minimum_fitness` stop evolving.")
        .default_value(1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--seed")
        .help("Seed to pass to random number generator.")
        .default_value(0)
        .action([](const std::string& value) { return std::stoul(value); });
    program.add_argument("--save_state")
        .help("Save state of DEAC algorithm. Saves the random number generator, population, and population fitness.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--save_directory")
        .help("Directory to save results in.")
        .default_value("./deacresults");
    program.add_argument("--track_stats")
        .help("Track minimum fitness and other stats.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--spectra_type")
        .help("Choose spectral type for kernel factors [bdsf, spbsf, spfsf, full].")
        .default_value("bdsf");
    program.add_argument("isf_file") //FIXME make this more generic
        .help("binary file containing isf data (tau, isf, error)");
    try {
      program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cout << err.what() << std::endl;
        std::cout << program << std::endl;
        exit(1);
    }

    std::random_device rd;
    auto seed_data = std::array<int, std::mt19937::state_size> {};
    std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    std::mt19937 generator(seq);
    uuids::uuid_random_generator gen{generator};
    
    uuids::uuid const id = gen();
    assert(!id.is_nil());
    assert(id.as_bytes().size() == 16);
    assert(id.version() == uuids::uuid_version::random_number_based);
    assert(id.variant() == uuids::uuid_variant::rfc);

    std::string uuid_str = uuids::to_string(id);
    std::cout << "uuid: " << uuid_str << std::endl;

    size_t number_of_elements;
    double* numpy_data;
    std::string isf_file = program.get<std::string>("isf_file");
    std::tie(numpy_data, number_of_elements) = load_numpy_array(isf_file);
    size_t number_of_timeslices = number_of_elements/3;

    double * const imaginary_time = numpy_data;
    double * const isf = numpy_data + number_of_timeslices;
    double * const isf_error = numpy_data + 2*number_of_timeslices;

    uint64_t seed = 1407513600 + static_cast<uint64_t>(program.get<unsigned long>("--seed"));
    uint64_t seed_int = static_cast<uint64_t>(program.get<unsigned long>("--seed"));
    struct xoshiro256p_state rng = xoshiro256p_init(seed);

    double temperature = program.get<double>("--temperature");
    #ifndef ZEROT
        assert(temperature > 0.0);
    #endif
    size_t number_of_generations = static_cast<size_t>(program.get<unsigned long>("--number_of_generations"));
    size_t population_size = static_cast<size_t>(program.get<unsigned long>("--population_size"));

    size_t genome_size;
    double * frequency;
    if (auto frequency_filename = program.present("--frequency_file")) {
        std::tie(frequency,genome_size) = load_numpy_array(*frequency_filename);
    } else{
        genome_size = static_cast<size_t>(program.get<unsigned long>("--genome_size"));
        double max_frequency = program.get<double>("--omega_max");

        frequency = (double*) malloc(sizeof(double)*genome_size);
        double dfrequency = max_frequency/(genome_size - 1);
        for (size_t i=0; i<genome_size; i++) {
            frequency[i] = i*dfrequency;
        }
    }

    bool normalize = program.get<bool>("--normalize");
    bool use_negative_first_moment = program.get<bool>("--use_negative_first_moment");
    double first_moment = program.get<double>("--first_moment");
    double third_moment = program.get<double>("--third_moment");
    double third_moment_error = program.get<double>("--third_moment_error");

    double crossover_probability = program.get<double>("--crossover_probability");
    double self_adapting_crossover_probability = program.get<double>("--self_adapting_crossover_probability");
    double differential_weight = program.get<double>("--differential_weight");
    double self_adapting_differential_weight_probability = program.get<double>("--self_adapting_differential_weight_probability");
    double self_adapting_differential_weight_shift = program.get<double>("--self_adapting_differential_weight_shift");
    double self_adapting_differential_weight = program.get<double>("--self_adapting_differential_weight");
    
    double stop_minimum_fitness = program.get<double>("--stop_minimum_fitness");

    bool track_stats = program.get<bool>("--track_stats");
    std::string save_directory_str = program.get<std::string>("--save_directory");
    fs::path save_directory(save_directory_str);
    fs::create_directory(save_directory);

    //Write to log file
    #ifndef SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION
        #ifndef ZEROT
            std::string deac_prefix = "deac";
        #endif
        #ifdef ZEROT
            std::string deac_prefix = "deac-zT";
        #endif
    #else
        #ifndef ZEROT
            std::string deac_prefix = "deac-spfsf";
        #endif
        #ifdef ZEROT
            std::string deac_prefix = "deac-zT-spfsf";
        #endif
    #endif
    std::string log_filename_str = string_format("%s_log_%s.dat",deac_prefix.c_str(),uuid_str.c_str());
    fs::path log_filename = save_directory / log_filename_str;
    std::ofstream log_ofs(log_filename.c_str(), std::ios_base::out | std::ios_base::app );

    //Input parameters
    log_ofs << "uuid: " << uuid_str << std::endl;
    log_ofs << "isf_file: " << isf_file << std::endl;
    log_ofs.close();

    //FIXME FIXME FIXME need to add this
    std::string spectra_type

    deac( &rng, imaginary_time, isf, isf_error, frequency, temperature,
            number_of_generations, number_of_timeslices, population_size, genome_size,
            normalize, use_negative_first_moment, first_moment, third_moment,
            third_moment_error, crossover_probability,
            self_adapting_crossover_probability, differential_weight,
            self_adapting_differential_weight_probability,
            self_adapting_differential_weight_shift,
            self_adapting_differential_weight, stop_minimum_fitness,
            track_stats, seed_int, uuid_str, spectra_type, save_directory);

    free(numpy_data);
    free(frequency);
    return 0;
}
