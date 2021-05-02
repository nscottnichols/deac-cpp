#include <stdio.h>
#include <math.h> // cosh
#include <stdlib.h>
#include <iostream>
#include <tuple> // for tie() and tuple
#include <argparse.hpp>
#include <rng.hpp>
#include <memory> // string_format
#include <string> // string_format
#include <stdexcept> // string_format
#include <fstream> // std::ofstream
#include <uuid.h>

#ifdef __GNUC__
    #if __GNUC__ > 7
        #include <filesystem>
        namespace fs = std::filesystem;
    #endif
    #if __GNUC__ < 8
        #include <experimental/filesystem>
        namespace fs = std::experimental::filesystem;
    #endif
#endif
#ifndef __GNUC__
    #include <filesystem>
    namespace fs = std::filesystem;
#endif

//GPU acceleration
#ifdef GPU_BLOCK_SIZE
    #ifndef USE_CUDA
        #include "deac_gpu.hip.hpp"
    #endif
    #ifdef USE_CUDA
        #include "deac_gpu.cuh"
    #endif
#endif

#ifdef DEAC_DEBUG
    #ifdef GPU_BLOCK_SIZE
        #ifndef USE_CUDA
            void h_gpu_check_array(double * _array, int length) {
                int grid_size = (length + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
                hipLaunchKernelGGL(gpu_check_array, (dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, 0, _array, length);
                HIP_ASSERT(hipDeviceSynchronize());
            }
        #endif
        #ifdef USE_CUDA
            void h_gpu_check_array(double * _array, int length) {
                int grid_size = (length + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
                cuda_wrapper::gpu_check_array_wrapper(dim3(grid_size), dim3(GPU_BLOCK_SIZE), _array, length);
                CUDA_ASSERT(cudaDeviceSynchronize());
            }
        #endif
    #endif
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

void write_array(fs::path filename, double * buffer, int length) {
    FILE * output_file;
    output_file = fopen (filename.c_str(), "wb");
    fwrite (buffer , sizeof(double), static_cast<size_t>(length), output_file);
    fclose (output_file);
}

std::tuple <double*, unsigned int> load_numpy_array(std::string isf_file) {
    FILE * input_file;
    long file_size_bytes;
    double * buffer;
    size_t result;
  
    input_file = fopen( isf_file.c_str(), "rb" );
    if (input_file==NULL) {fputs("File error",stderr); exit(1);}
  
    // obtain file size:
    fseek(input_file , 0 , SEEK_END);
    file_size_bytes = ftell(input_file);
    rewind(input_file);
    
    unsigned int number_of_elements = static_cast<unsigned int> (file_size_bytes/sizeof(double));
  
    // allocate memory to contain the whole file:
    buffer = (double*) malloc(sizeof(char)*file_size_bytes);
    if (buffer == NULL) {fputs("Memory error",stderr); exit(2);}
  
    // copy the file into the buffer:
    result = fread(buffer,1,file_size_bytes,input_file);
    if (result != file_size_bytes) {fputs("Reading error",stderr); exit(3);}
  
    /* the whole file is now loaded in the memory buffer. */
    fclose (input_file);

    std::tuple <double*,unsigned int> numpy_data_tuple(buffer, number_of_elements);
    return numpy_data_tuple;
}

void matrix_multiply_MxN_by_Nx1(double * C, double * A, double * B, int M, int N) {
    for (int i=0; i<M; i++) {
        C[i] = 0.0;
        for (int j=0; j<N; j++) {
            C[i] += A[i*N + j]*B[j];
        }
    }
}

void matrix_multiply_LxM_by_MxN(double * C, double * A, double * B, int L,
        int M, int N) {
    for (int i=0; i<N; i++) {
        for (int j=0; j<L; j++) {
            C[i*L + j] = 0.0;
            for (int k=0; k<M; k++) {
                C[i*L + j] += A[j*M + k]*B[i*M + k];
            }
        }
    }
}

double reduced_chi_square_statistic(double * observed, double * calculated,
        double * error, int length) {
    double chi_squared = 0.0;
    for (int i=0; i<length; i++) {
        chi_squared += pow((observed[i] - calculated[i])/error[i],2);
    }
    return chi_squared;
}

double minimum(double * A, int length) {
    double _minimum = A[0];
    for (int i=0; i<length; i++) {
        if (A[i] < _minimum) {
            _minimum = A[i];
        }
    }
    return _minimum;
}

int argmin(double * A, int length) {
    int _argmin=0;
    double _minimum = A[0];
    for (int i=0; i<length; i++) {
        if (A[i] < _minimum) {
            _minimum = A[i];
            _argmin = i;
        }
    }
    return _argmin;
}

std::tuple <int, double> argmin_and_min(double * A, int length) {
    int _argmin=0;
    double _minimum = A[0];
    for (int i=0; i<length; i++) {
        if (A[i] < _minimum) {
            _minimum = A[i];
            _argmin = i;
        }
    }
    std::tuple <int, double> argmin_tuple(_argmin, _minimum);
    return argmin_tuple;
}

double mean(double * A, int length) {
    double _mean = 0.0;
    for (int i=0; i<length; i++) {
        _mean += A[i];
    }
    return _mean/length;
}

double standard_deviation(double * A, double mean, int length) {
    double _std = 0.0;
    for (int i=0; i<length; i++) {
        _std += pow(A[i] - mean,2);
    }
    return sqrt(_std/length);
}

std::tuple <int, int, int> get_mutant_indices(struct xoshiro256p_state * rng,
        int mutant_index0, int length) {
    int mutant_index1 = mutant_index0;
    int mutant_index2 = mutant_index0;
    int mutant_index3 = mutant_index0;
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

    std::tuple <int, int, int> _mutant_indices(mutant_index1, mutant_index2, mutant_index3);
    return _mutant_indices;
}

void set_mutant_indices(struct xoshiro256p_state * rng, int * mutant_indices,
        int mutant_index0, int length) {
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
        double temperature, int number_of_generations, int number_of_timeslices, int population_size,
        int genome_size, bool normalize, bool use_inverse_first_moment, 
        double first_moment, double third_moment, double third_moment_error,
        double crossover_probability,
        double self_adapting_crossover_probability,
        double differential_weight, 
        double self_adapting_differential_weight_probability,
        double self_adapting_differential_weight_shift,
        double self_adapting_differential_weight, double stop_minimum_fitness,
        bool track_stats, int seed, std::string uuid_str, fs::path save_directory) {

    //Create GPU device streams
    #ifdef GPU_BLOCK_SIZE
        #ifndef USE_CUDA
            hipStream_t stream_array[MAX_GPU_STREAMS];
        #endif
        #ifdef USE_CUDA
            cudaStream_t stream_array[MAX_GPU_STREAMS];
        #endif
        for (int i = 0; i < MAX_GPU_STREAMS; i++) {
            #ifndef USE_CUDA
                HIP_ASSERT(hipStreamCreate(&stream_array[i]));
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaStreamCreate(&stream_array[i]));
            #endif
        }
    #endif

    #ifdef GPU_BLOCK_SIZE
        double * d_isf;
        double * d_isf_error;
        size_t bytes_isf = sizeof(double)*number_of_timeslices;
        size_t bytes_isf_error = sizeof(double)*number_of_timeslices;
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_isf, bytes_isf));
            HIP_ASSERT(hipMalloc(&d_isf_error, bytes_isf_error));
            HIP_ASSERT(hipMemcpy( d_isf, isf, bytes_isf, hipMemcpyHostToDevice ));
            HIP_ASSERT(hipMemcpy( d_isf_error, isf_error, bytes_isf_error, hipMemcpyHostToDevice ));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_isf, bytes_isf));
            CUDA_ASSERT(cudaMalloc(&d_isf_error, bytes_isf_error));
            CUDA_ASSERT(cudaMemcpy( d_isf, isf, bytes_isf, cudaMemcpyHostToDevice ));
            CUDA_ASSERT(cudaMemcpy( d_isf_error, isf_error, bytes_isf_error, cudaMemcpyHostToDevice ));
        #endif
    #endif

    double beta = 1.0/temperature;
    double zeroth_moment = isf[0];
    bool use_first_moment = first_moment >= 0.0;
    bool use_third_moment = third_moment >= 0.0;

    double * isf_term;
    size_t bytes_isf_term = sizeof(double)*genome_size*number_of_timeslices;
    isf_term = (double*) malloc(bytes_isf_term);
    for (int i=0; i<number_of_timeslices; i++) {
        double t = imaginary_time[i];
        double bo2mt = 0.5*beta - t;
        for (int j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            int isf_term_idx = i*genome_size + j;
            isf_term[isf_term_idx] = df*cosh(bo2mt*f);
        }
    }
    
    #ifdef GPU_BLOCK_SIZE
        double * d_isf_term; // pointer to isf_term on gpu
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_isf_term, bytes_isf_term)); // Allocate memory for isf_term on GPU
            HIP_ASSERT(hipMemcpy( d_isf_term, isf_term, bytes_isf_term, hipMemcpyHostToDevice )); // Copy isf_term data to gpu
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_isf_term, bytes_isf_term)); // Allocate memory for isf_term on GPU
            CUDA_ASSERT(cudaMemcpy( d_isf_term, isf_term, bytes_isf_term, cudaMemcpyHostToDevice )); // Copy isf_term data to gpu
        #endif
    #endif

    //Generate population and set initial fitness
    double * population_old;
    double * population_new;
    size_t bytes_population = sizeof(double)*genome_size*population_size;
    population_old = (double*) malloc(bytes_population);
    population_new = (double*) malloc(bytes_population);
    for (int i=0; i<genome_size*population_size; i++) {
        population_old[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53; // to_double2
    }

    #ifdef GPU_BLOCK_SIZE
        double * d_population_old;
        double * d_population_new;
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_population_old, bytes_population));
            HIP_ASSERT(hipMalloc(&d_population_new, bytes_population));
            HIP_ASSERT(hipMemcpy( d_population_old, population_old, bytes_population, hipMemcpyHostToDevice ));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_population_old, bytes_population));
            CUDA_ASSERT(cudaMalloc(&d_population_new, bytes_population));
            CUDA_ASSERT(cudaMemcpy( d_population_old, population_old, bytes_population, cudaMemcpyHostToDevice )); 
        #endif
    #endif

    // Normalize population
    double * normalization_term;
    double * normalization;
    #ifdef GPU_BLOCK_SIZE
        double * d_normalization_term;
        double * d_normalization;
    #endif
    size_t bytes_normalization_term = sizeof(double)*genome_size;
    size_t bytes_normalization = sizeof(double)*population_size;
    if (normalize) {
        normalization_term = (double*) malloc(bytes_normalization_term);
        for (int j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            normalization_term[j] = df*cosh(0.5*beta*f);
        }
        normalization = (double*) malloc(bytes_normalization);
        #ifdef GPU_BLOCK_SIZE
            #ifndef USE_CUDA
                HIP_ASSERT(hipMalloc(&d_normalization_term, bytes_normalization_term));
                HIP_ASSERT(hipMalloc(&d_normalization, bytes_normalization));
                HIP_ASSERT(hipMemcpy( d_normalization_term, normalization_term, bytes_normalization_term, hipMemcpyHostToDevice ));
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMalloc(&d_normalization_term, bytes_normalization_term));
                CUDA_ASSERT(cudaMalloc(&d_normalization, bytes_normalization));
                CUDA_ASSERT(cudaMemcpy( d_normalization_term, normalization_term, bytes_normalization_term, cudaMemcpyHostToDevice )); 
            #endif
        #endif

        #ifdef GPU_BLOCK_SIZE
            int grid_size_set_normalization = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            #ifndef USE_CUDA
                HIP_ASSERT(hipMemset(d_normalization,0, bytes_normalization));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    hipLaunchKernelGGL(gpu_matrix_multiply_MxN_by_Nx1,
                            dim3(grid_size_set_normalization), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                            d_normalization, d_population_old, d_normalization_term, genome_size, i);
                }
                HIP_ASSERT(hipDeviceSynchronize());
                int grid_size_normalize_population = (population_size*genome_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
                hipLaunchKernelGGL(gpu_normalize_population, dim3(grid_size_normalize_population), dim3(GPU_BLOCK_SIZE), 0, 0,
                        d_population_old, d_normalization, zeroth_moment, population_size, genome_size); 
                HIP_ASSERT(hipDeviceSynchronize());
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMemset(d_normalization,0, bytes_normalization));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    cuda_wrapper::gpu_matrix_multiply_MxN_by_Nx1_wrapper(
                            dim3(grid_size_set_normalization), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                            d_normalization, d_population_old, d_normalization_term, genome_size, i);
                }
                CUDA_ASSERT(cudaDeviceSynchronize());

                matrix_multiply_MxN_by_Nx1(normalization, population_old,
                        normalization_term, population_size, genome_size);
                int grid_size_normalize_population = (population_size*genome_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
                cuda_wrapper::gpu_normalize_population_wrapper( dim3(grid_size_normalize_population), dim3(GPU_BLOCK_SIZE),
                        d_population_old, d_normalization, zeroth_moment, population_size, genome_size); 
                CUDA_ASSERT(cudaDeviceSynchronize());
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
            matrix_multiply_MxN_by_Nx1(normalization, population_old,
                    normalization_term, population_size, genome_size);
            for (int i=0; i<population_size; i++) {
                double _norm = normalization[i];
                for (int j=0; j<genome_size; j++) {
                    population_old[i*genome_size + j] *= zeroth_moment/_norm;
                }
            }
        #endif

    }

    double * first_moments_term;
    double * first_moments;
    #ifdef GPU_BLOCK_SIZE
        double * d_first_moments_term;
        double * d_first_moments;
    #endif
    size_t bytes_first_moments_term = sizeof(double)*genome_size;
    size_t bytes_first_moments = sizeof(double)*population_size;
    if (use_first_moment) {
        first_moments_term = (double*) malloc(bytes_first_moments_term);
        for (int j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            first_moments_term[j] = df*f*sinh(0.5*beta*f);
        }

        first_moments = (double*) malloc(bytes_first_moments);
        #ifdef GPU_BLOCK_SIZE
            #ifndef USE_CUDA
                HIP_ASSERT(hipMalloc(&d_first_moments_term, bytes_first_moments_term));
                HIP_ASSERT(hipMalloc(&d_first_moments, bytes_first_moments));
                HIP_ASSERT(hipMemcpy( d_first_moments_term, first_moments_term, bytes_first_moments_term, hipMemcpyHostToDevice ));
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMalloc(&d_first_moments_term, bytes_first_moments_term));
                CUDA_ASSERT(cudaMalloc(&d_first_moments, bytes_first_moments));
                CUDA_ASSERT(cudaMemcpy( d_first_moments_term, first_moments_term, bytes_first_moments_term, cudaMemcpyHostToDevice )); 
            #endif
        #endif
        #ifdef GPU_BLOCK_SIZE
            int grid_size_set_first_moments = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            #ifndef USE_CUDA
                HIP_ASSERT(hipMemset(d_first_moments,0, bytes_first_moments));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    hipLaunchKernelGGL(gpu_matrix_multiply_MxN_by_Nx1,
                            dim3(grid_size_set_first_moments), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                            d_first_moments, d_population_old, d_first_moments_term, genome_size, i);
                }
                HIP_ASSERT(hipDeviceSynchronize());
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMemset(d_first_moments,0, bytes_first_moments));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    cuda_wrapper::gpu_matrix_multiply_MxN_by_Nx1_wrapper(
                            dim3(grid_size_set_first_moments), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                            d_first_moments, d_population_old, d_first_moments_term, genome_size, i);
                }
                CUDA_ASSERT(cudaDeviceSynchronize());
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
            matrix_multiply_MxN_by_Nx1(first_moments, population_old,
                    first_moments_term, population_size, genome_size);
        #endif
    }

    double * third_moments_term;
    double * third_moments;
    #ifdef GPU_BLOCK_SIZE
        double * d_third_moments_term;
        double * d_third_moments;
    #endif
    size_t bytes_third_moments_term = sizeof(double)*genome_size;
    size_t bytes_third_moments = sizeof(double)*population_size;
    if (use_third_moment) {
        third_moments_term = (double*) malloc(bytes_third_moments_term);
        for (int j=0; j<genome_size; j++) {
            double f = frequency[j];
            double df;
            if (j==0) {
                df = 0.5*(frequency[j+1] - frequency[j]);
            } else if (j == genome_size - 1) {
                df = 0.5*(frequency[j] - frequency[j-1]);
            } else {
                df = 0.5*(frequency[j+1] - frequency[j-1]);
            }
            third_moments_term[j] = df*pow(f,3)*sinh(0.5*beta*f);
        }

        third_moments = (double*) malloc(bytes_third_moments);
        #ifdef GPU_BLOCK_SIZE
            #ifndef USE_CUDA
                HIP_ASSERT(hipMalloc(&d_third_moments_term, bytes_third_moments_term));
                HIP_ASSERT(hipMalloc(&d_third_moments, bytes_third_moments));
                HIP_ASSERT(hipMemcpy( d_third_moments_term, third_moments_term, bytes_third_moments_term, hipMemcpyHostToDevice ));
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMalloc(&d_third_moments_term, bytes_third_moments_term));
                CUDA_ASSERT(cudaMalloc(&d_third_moments, bytes_third_moments));
                CUDA_ASSERT(cudaMemcpy( d_third_moments_term, third_moments_term, bytes_third_moments_term, cudaMemcpyHostToDevice )); 
            #endif
        #endif
        #ifdef GPU_BLOCK_SIZE
            int grid_size_set_third_moments = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            #ifndef USE_CUDA
                HIP_ASSERT(hipMemset(d_third_moments,0, bytes_third_moments));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    hipLaunchKernelGGL(gpu_matrix_multiply_MxN_by_Nx1,
                            dim3(grid_size_set_third_moments), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                            d_third_moments, d_population_old, d_third_moments_term, genome_size, i);
                }
                HIP_ASSERT(hipDeviceSynchronize());
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMemset(d_third_moments,0, bytes_third_moments));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    cuda_wrapper::gpu_matrix_multiply_MxN_by_Nx1_wrapper(
                            dim3(grid_size_set_third_moments), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                            d_third_moments, d_population_old, d_third_moments_term, genome_size, i);
                }
                CUDA_ASSERT(cudaDeviceSynchronize());
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
            matrix_multiply_MxN_by_Nx1(third_moments, population_old,
                    third_moments_term, population_size, genome_size);
        #endif
    }

    //set isf_model and calculate fitness
    double * isf_model;
    #ifdef GPU_BLOCK_SIZE
        double * d_isf_model;
    #endif
    size_t bytes_isf_model = sizeof(double)*number_of_timeslices*population_size;
    isf_model = (double*) malloc(bytes_isf_model);
    #ifdef GPU_BLOCK_SIZE
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_isf_model, bytes_isf_model));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_isf_model, bytes_isf_model));
        #endif
    #endif
    #ifdef GPU_BLOCK_SIZE
        int grid_size_set_isf_model = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
        #ifndef USE_CUDA
            HIP_ASSERT(hipMemset(d_isf_model,0, bytes_isf_model));
            for (int i=0; i<population_size*number_of_timeslices; i++) {
                int stream_idx = i % MAX_GPU_STREAMS;
                hipLaunchKernelGGL(gpu_matrix_multiply_LxM_by_MxN,
                        dim3(grid_size_set_isf_model), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                        d_isf_model, d_isf_term, d_population_old, number_of_timeslices, genome_size, i);
            }
            HIP_ASSERT(hipDeviceSynchronize());
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMemset(d_isf_model,0, bytes_isf_model));
            for (int i=0; i<population_size*number_of_timeslices; i++) {
                int stream_idx = i % MAX_GPU_STREAMS;
                cuda_wrapper::gpu_matrix_multiply_LxM_by_MxN_wrapper(
                        dim3(grid_size_set_isf_model), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                        d_isf_model, d_isf_term, d_population_old, number_of_timeslices, genome_size, i);
            }
            CUDA_ASSERT(cudaDeviceSynchronize());
        #endif
    #endif
    #ifndef GPU_BLOCK_SIZE
        matrix_multiply_LxM_by_MxN(isf_model, isf_term, population_old,
                number_of_timeslices, genome_size, population_size);
    #endif

    double * inverse_first_moments_term;
    double * inverse_first_moments;
    #ifdef GPU_BLOCK_SIZE
        double * d_inverse_first_moments_term;
        double * d_inverse_first_moments;
    #endif
    size_t bytes_inverse_first_moments_term = sizeof(double)*number_of_timeslices;
    size_t bytes_inverse_first_moments = sizeof(double)*population_size;
    double inverse_first_moment = 0.0;
    double inverse_first_moment_error = 0.0;
    if (use_inverse_first_moment) {
        inverse_first_moments_term = (double*) malloc(bytes_inverse_first_moments_term);
        for (int j=0; j<number_of_timeslices; j++) {
            double dt;
            if (j==0) {
                dt = 0.5*(imaginary_time[j+1] - imaginary_time[j]);
            } else if (j == number_of_timeslices - 1) {
                dt = 0.5*(imaginary_time[j] - imaginary_time[j-1]);
            } else {
                dt = 0.5*(imaginary_time[j+1] - imaginary_time[j-1]);
            }
            inverse_first_moments_term[j] = dt;
            inverse_first_moment += isf[j]*dt;
            inverse_first_moment_error = pow(isf_error[j],2) * pow(dt,2);
        }
        inverse_first_moment_error = sqrt(inverse_first_moment_error);

        inverse_first_moments = (double*) malloc(bytes_inverse_first_moments);
        #ifdef GPU_BLOCK_SIZE
            #ifndef USE_CUDA
                HIP_ASSERT(hipMalloc(&d_inverse_first_moments_term, bytes_inverse_first_moments_term));
                HIP_ASSERT(hipMalloc(&d_inverse_first_moments, bytes_inverse_first_moments));
                HIP_ASSERT(hipMemcpy( d_inverse_first_moments_term, inverse_first_moments_term, bytes_inverse_first_moments_term, hipMemcpyHostToDevice ));
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMalloc(&d_inverse_first_moments_term, bytes_inverse_first_moments_term));
                CUDA_ASSERT(cudaMalloc(&d_inverse_first_moments, bytes_inverse_first_moments));
                CUDA_ASSERT(cudaMemcpy( d_inverse_first_moments_term, inverse_first_moments_term, bytes_inverse_first_moments_term, cudaMemcpyHostToDevice )); 
            #endif
        #endif
        #ifdef GPU_BLOCK_SIZE
            int grid_size_set_inverse_first_moments = (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            #ifndef USE_CUDA
                HIP_ASSERT(hipMemset(d_inverse_first_moments,0, bytes_inverse_first_moments));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    hipLaunchKernelGGL(gpu_matrix_multiply_MxN_by_Nx1,
                            dim3(grid_size_set_inverse_first_moments), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                            d_inverse_first_moments, d_isf_model, d_inverse_first_moments_term, number_of_timeslices, i);
                }
                HIP_ASSERT(hipDeviceSynchronize());
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMemset(d_inverse_first_moments,0, bytes_inverse_first_moments));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    cuda_wrapper::gpu_matrix_multiply_MxN_by_Nx1_wrapper(
                            dim3(grid_size_set_inverse_first_moments), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                            d_inverse_first_moments, d_isf_model, d_inverse_first_moments_term, number_of_timeslices, i);
                }
                CUDA_ASSERT(cudaDeviceSynchronize());
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
            matrix_multiply_MxN_by_Nx1(inverse_first_moments, isf_model,
                    inverse_first_moments_term, population_size, number_of_timeslices);
        #endif
    }

    double * fitness_old;
    #ifdef GPU_BLOCK_SIZE
        double * d_fitness_old;
        double * d_fitness_new;
        size_t bytes_fitness_new = sizeof(double)*population_size;
    #endif
    size_t bytes_fitness_old = sizeof(double)*population_size;
    fitness_old = (double*) malloc(bytes_fitness_old);

    #ifdef GPU_BLOCK_SIZE
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_fitness_old, bytes_fitness_old));
            HIP_ASSERT(hipMalloc(&d_fitness_new, bytes_fitness_new));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_fitness_old, bytes_fitness_old));
            CUDA_ASSERT(cudaMalloc(&d_fitness_new, bytes_fitness_new));
        #endif
    #endif
    #ifdef GPU_BLOCK_SIZE
        int grid_size_set_fitness = (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
        int grid_size_set_fitness_moments = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
        #ifndef USE_CUDA
            HIP_ASSERT(hipMemset(d_fitness_old,0, bytes_fitness_old));
            for (int i=0; i<population_size; i++) {
                int stream_idx = i % MAX_GPU_STREAMS;
                hipLaunchKernelGGL(gpu_set_fitness,
                        dim3(grid_size_set_fitness), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                        d_fitness_old, d_isf, d_isf_model, d_isf_error, number_of_timeslices, i);
            }
            HIP_ASSERT(hipDeviceSynchronize());
            if (use_inverse_first_moment) {
                hipLaunchKernelGGL(gpu_set_fitness_moments_reduced_chi_squared,
                        dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE), 0, 0,
                        d_fitness_old, d_inverse_first_moments, inverse_first_moment, inverse_first_moment_error, population_size);
                HIP_ASSERT(hipDeviceSynchronize());
            }
            if (use_first_moment) {
                hipLaunchKernelGGL(gpu_set_fitness_moments_chi_squared,
                        dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE), 0, 0,
                        d_fitness_old, d_first_moments, first_moment, population_size);
                HIP_ASSERT(hipDeviceSynchronize());
            }
            if (use_third_moment) {
                hipLaunchKernelGGL(gpu_set_fitness_moments_reduced_chi_squared,
                        dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE), 0, 0,
                        d_fitness_old, d_third_moments, third_moment, third_moment_error, population_size);
                HIP_ASSERT(hipDeviceSynchronize());
            }
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMemset(d_fitness_old,0, bytes_fitness_old));
            for (int i=0; i<population_size; i++) {
                int stream_idx = i % MAX_GPU_STREAMS;
                cuda_wrapper::gpu_set_fitness_wrapper(
                        dim3(grid_size_set_fitness), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                        d_fitness_old, d_isf, d_isf_model, d_isf_error, number_of_timeslices, i);
            }
            CUDA_ASSERT(cudaDeviceSynchronize());
            if (use_inverse_first_moment) {
                cuda_wrapper::gpu_set_fitness_moments_reduced_chi_squared_wrapper(
                        dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE),
                        d_fitness_old, d_inverse_first_moments, inverse_first_moment, inverse_first_moment_error, population_size);
                CUDA_ASSERT(cudaDeviceSynchronize());
            }
            if (use_first_moment) {
                cuda_wrapper::gpu_set_fitness_moments_chi_squared_wrapper(
                        dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE),
                        d_fitness_old, d_first_moments, first_moment, population_size);
                CUDA_ASSERT(cudaDeviceSynchronize());
            }
            if (use_third_moment) {
                cuda_wrapper::gpu_set_fitness_moments_reduced_chi_squared_wrapper(
                        dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE),
                        d_fitness_old, d_third_moments, third_moment, third_moment_error, population_size);
                CUDA_ASSERT(cudaDeviceSynchronize());
            }
        #endif
    #endif
    #ifndef GPU_BLOCK_SIZE
        for (int i=0; i<population_size; i++) {
            double _fitness = reduced_chi_square_statistic(isf,
                    isf_model + i*number_of_timeslices, isf_error,
                    number_of_timeslices)/number_of_timeslices;
            if (use_inverse_first_moment) {
                _fitness += pow((inverse_first_moment - inverse_first_moments[i])/inverse_first_moment_error,2);
            }
            if (use_first_moment) {
                _fitness += pow(first_moments[i] - first_moment,2)/first_moment;
            }
            if (use_third_moment) {
                _fitness += pow((third_moment - third_moments[i])/third_moment_error,2);
            }
            fitness_old[i] = _fitness;
        }
    #endif

    double * crossover_probabilities_old;
    double * crossover_probabilities_new;
    size_t bytes_crossover_probabilities = sizeof(double)*population_size;
    crossover_probabilities_old = (double*) malloc(bytes_crossover_probabilities);
    crossover_probabilities_new = (double*) malloc(bytes_crossover_probabilities);
    for (int i=0; i<population_size; i++) {
        crossover_probabilities_old[i] = crossover_probability;
    }
    #ifdef GPU_BLOCK_SIZE
        double * d_crossover_probabilities_old;
        double * d_crossover_probabilities_new;
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_crossover_probabilities_old, bytes_crossover_probabilities));
            HIP_ASSERT(hipMalloc(&d_crossover_probabilities_new, bytes_crossover_probabilities));
            HIP_ASSERT(hipMemcpy( d_crossover_probabilities_old, crossover_probabilities_old, bytes_crossover_probabilities, hipMemcpyHostToDevice ));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_crossover_probabilities_old, bytes_crossover_probabilities));
            CUDA_ASSERT(cudaMalloc(&d_crossover_probabilities_new, bytes_crossover_probabilities));
            CUDA_ASSERT(cudaMemcpy( d_crossover_probabilities_old, crossover_probabilities_old, bytes_crossover_probabilities, cudaMemcpyHostToDevice )); 
        #endif
    #endif

    double * differential_weights_old;
    double * differential_weights_new;
    size_t bytes_differential_weights = sizeof(double)*population_size;
    differential_weights_old = (double*) malloc(bytes_differential_weights);
    differential_weights_new = (double*) malloc(bytes_differential_weights);
    for (int i=0; i<population_size; i++) {
        differential_weights_old[i] = differential_weight;
    }
    #ifdef GPU_BLOCK_SIZE
        double * d_differential_weights_old;
        double * d_differential_weights_new;
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_differential_weights_old, bytes_differential_weights));
            HIP_ASSERT(hipMalloc(&d_differential_weights_new, bytes_differential_weights));
            HIP_ASSERT(hipMemcpy( d_differential_weights_old, differential_weights_old, bytes_differential_weights, hipMemcpyHostToDevice ));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_differential_weights_old, bytes_differential_weights));
            CUDA_ASSERT(cudaMalloc(&d_differential_weights_new, bytes_differential_weights));
            CUDA_ASSERT(cudaMemcpy( d_differential_weights_old, differential_weights_old, bytes_differential_weights, cudaMemcpyHostToDevice )); 
        #endif
    #endif

    //Initialize statistics arrays
    double * fitness_mean;
    double * fitness_minimum;
    double * fitness_standard_deviation;
    #ifdef GPU_BLOCK_SIZE
        double * d_fitness_mean;
        double * d_fitness_minimum;
        double * d_fitness_standard_deviation;
    #endif
    size_t bytes_fitness_mean = sizeof(double)*number_of_generations;
    size_t bytes_fitness_minimum = sizeof(double)*number_of_generations;
    size_t bytes_fitness_standard_deviation = sizeof(double)*number_of_generations;
    if (track_stats) {
        fitness_mean = (double*) malloc(bytes_fitness_mean);
        fitness_minimum = (double*) malloc(bytes_fitness_minimum);
        fitness_standard_deviation = (double*) malloc(bytes_fitness_standard_deviation);
        #ifdef GPU_BLOCK_SIZE
            #ifndef USE_CUDA
                HIP_ASSERT(hipMalloc(&d_fitness_mean, bytes_fitness_mean));
                HIP_ASSERT(hipMalloc(&d_fitness_minimum, bytes_fitness_minimum));
                HIP_ASSERT(hipMalloc(&d_fitness_standard_deviation, bytes_fitness_standard_deviation));
                HIP_ASSERT(hipMemset(d_fitness_mean,0, bytes_fitness_mean));
                HIP_ASSERT(hipMemset(d_fitness_minimum,0, bytes_fitness_minimum));
                HIP_ASSERT(hipMemset(d_fitness_standard_deviation,0, bytes_fitness_standard_deviation));
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMalloc(&d_fitness_mean, bytes_fitness_mean));
                CUDA_ASSERT(cudaMalloc(&d_fitness_minimum, bytes_fitness_minimum));
                CUDA_ASSERT(cudaMalloc(&d_fitness_standard_deviation, bytes_fitness_standard_deviation));
                CUDA_ASSERT(cudaMemset(d_fitness_mean,0, bytes_fitness_mean));
                CUDA_ASSERT(cudaMemset(d_fitness_minimum,0, bytes_fitness_minimum));
                CUDA_ASSERT(cudaMemset(d_fitness_standard_deviation,0, bytes_fitness_standard_deviation));
            #endif
        #endif
    }
    
    bool * mutate_indices;
    #ifdef GPU_BLOCK_SIZE
        bool * d_mutate_indices;
        bool * d_rejection_indices;
        size_t bytes_rejection_indices = sizeof(bool)*population_size;
    #endif
    size_t bytes_mutate_indices = sizeof(bool)*genome_size*population_size;
    mutate_indices = (bool*) malloc(bytes_mutate_indices);
    #ifdef GPU_BLOCK_SIZE
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_mutate_indices, bytes_mutate_indices));
            HIP_ASSERT(hipMalloc(&d_rejection_indices, bytes_rejection_indices));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_mutate_indices, bytes_mutate_indices));
            CUDA_ASSERT(cudaMalloc(&d_rejection_indices, bytes_rejection_indices));
        #endif
    #endif

    int * mutant_indices;
    #ifdef GPU_BLOCK_SIZE
        int * d_mutant_indices;
    #endif
    size_t bytes_mutant_indices = sizeof(int)*3*population_size;
    mutant_indices = (int*) malloc(bytes_mutant_indices);
    #ifdef GPU_BLOCK_SIZE
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_mutant_indices, bytes_mutant_indices));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_mutant_indices, bytes_mutant_indices));
        #endif
    #endif

    double minimum_fitness;
    int minimum_fitness_idx;
    #ifdef GPU_BLOCK_SIZE
        double * d_minimum_fitness;
        size_t bytes_minimum_fitness = sizeof(double);
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_minimum_fitness, bytes_minimum_fitness));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_minimum_fitness, bytes_minimum_fitness));
        #endif
    #endif

    #ifdef GPU_BLOCK_SIZE
        // Generate rng state
        uint64_t * rng_state;
        uint64_t * d_rng_state;
        size_t bytes_rng_state = sizeof(uint64_t)*4*population_size*(genome_size + 1);
        rng_state = (uint64_t *) malloc(bytes_rng_state);

        for (int i=0; i<population_size; i++) {
            for (int j=0; j < genome_size + 1; j++) {
                xoshiro256p_copy_state(rng_state + 4*(i*genome_size + j), rng->s);
                xoshiro256p_jump(rng->s);
            }
        }
        #ifndef USE_CUDA
            HIP_ASSERT(hipMalloc(&d_rng_state, bytes_rng_state));
            HIP_ASSERT(hipMemcpy( d_rng_state, rng_state, bytes_rng_state, hipMemcpyHostToDevice ));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaMalloc(&d_rng_state, bytes_rng_state));
            CUDA_ASSERT(cudaMemcpy( d_rng_state, rng_state, bytes_rng_state, cudaMemcpyHostToDevice ));
        #endif
    #endif
    
    int generation;
    for (int ii=0; ii < number_of_generations - 1; ii++) {
        generation = ii;
        #ifdef GPU_BLOCK_SIZE
            #ifndef USE_CUDA
                hipLaunchKernelGGL(gpu_get_minimum_fitness,
                        dim3(1), dim3(GPU_BLOCK_SIZE), 0, 0,
                        d_fitness_old, d_minimum_fitness, population_size);
                HIP_ASSERT(hipDeviceSynchronize());
                hipMemcpy(&minimum_fitness, d_minimum_fitness, bytes_minimum_fitness, hipMemcpyDeviceToHost);
            #endif
            #ifdef USE_CUDA
                cuda_wrapper::gpu_get_minimum_fitness_wrapper(
                        dim3(1), dim3(GPU_BLOCK_SIZE),
                        d_fitness_old, d_minimum_fitness, population_size);
                CUDA_ASSERT(cudaDeviceSynchronize());
                cudaMemcpy(&minimum_fitness, d_minimum_fitness, bytes_minimum_fitness, cudaMemcpyDeviceToHost);
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
            minimum_fitness = minimum(fitness_old,population_size);
        #endif

        //FIXME experimental break loop without transferring memory
        //#ifdef GPU_BLOCK_SIZE
        //    #ifndef USE_CUDA
        //        hipLaunchKernelGGL(gpu_check_minimum_fitness,
        //                dim3(1), dim3(1), 0, 0,
        //                d_minimum_fitness, stop_minimum_fitness
        //                );
        //        if (hipPeekAtLastError() != hipSuccess) {
        //            break;
        //        }
        //    #endif
        //    #ifdef USE_CUDA
        //        cuda_wrapper::gpu_check_minimum_fitness_wrapper(
        //                dim3(1), dim3(1),
        //                d_minimum_fitness, stop_minimum_fitness
        //                );
        //        if (cudaPeekAtLastError() != cudaSuccess) {
        //            break;
        //        }
        //    #endif
        //#endif
        //#ifndef GPU_BLOCK_SIZE
        ////Stopping criteria
        //if (minimum_fitness <= stop_minimum_fitness) {
        //    break;
        //}
        //#endif

        //Get Statistics
        if (track_stats) {
            #ifdef GPU_BLOCK_SIZE
                int grid_size_set_fitness_mean = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                int grid_size_set_fitness_standard_deviation = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                #ifndef USE_CUDA
                    hipLaunchKernelGGL(gpu_set_fitness_mean,
                            dim3(grid_size_set_fitness_mean), dim3(GPU_BLOCK_SIZE), 0, 0,
                            d_fitness_mean, d_fitness_old, population_size, ii);
                    HIP_ASSERT(hipDeviceSynchronize());
                    hipMemcpy(d_fitness_minimum + ii, d_minimum_fitness, bytes_minimum_fitness, hipMemcpyDeviceToDevice);
                    hipLaunchKernelGGL(gpu_set_fitness_standard_deviation,
                            dim3(grid_size_set_fitness_standard_deviation), dim3(GPU_BLOCK_SIZE), 0, 0,
                            d_fitness_standard_deviation, d_fitness_mean, d_fitness_old, population_size, ii);
                    HIP_ASSERT(hipDeviceSynchronize());
                #endif
                #ifdef USE_CUDA
                    cuda_wrapper::gpu_set_fitness_mean_wrapper(
                            dim3(grid_size_set_fitness_mean), dim3(GPU_BLOCK_SIZE),
                            d_fitness_mean, d_fitness_old, population_size, ii);
                    CUDA_ASSERT(cudaDeviceSynchronize());
                    cudaMemcpy(d_fitness_minimum + ii, d_minimum_fitness, bytes_minimum_fitness, cudaMemcpyDeviceToDevice);
                    cuda_wrapper::gpu_set_fitness_standard_deviation_wrapper(
                            dim3(grid_size_set_fitness_standard_deviation), dim3(GPU_BLOCK_SIZE),
                            d_fitness_standard_deviation, d_fitness_mean, d_fitness_old, population_size, ii);
                    CUDA_ASSERT(cudaDeviceSynchronize());
                #endif
            #endif
            #ifndef GPU_BLOCK_SIZE
                fitness_mean[ii] = mean(fitness_old, population_size);
                fitness_minimum[ii] = minimum_fitness;
                fitness_standard_deviation[ii] = standard_deviation(fitness_old,
                        fitness_mean[ii], population_size);
            #endif
        }
        
        //Stopping criteria
        if (minimum_fitness <= stop_minimum_fitness) {
            break;
        }


        #ifdef GPU_BLOCK_SIZE
            int grid_size_self_adapting_parameters = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            #ifndef USE_CUDA
                hipLaunchKernelGGL(gpu_set_crossover_probabilities_new,
                        dim3(grid_size_self_adapting_parameters), dim3(GPU_BLOCK_SIZE), 0, stream_array[0],
                        d_rng_state, d_crossover_probabilities_new, d_crossover_probabilities_old, self_adapting_crossover_probability, population_size);
                hipLaunchKernelGGL(gpu_set_differential_weights_new,
                        dim3(grid_size_self_adapting_parameters), dim3(GPU_BLOCK_SIZE), 0, stream_array[1 % MAX_GPU_STREAMS],
                        d_rng_state + 4*population_size, d_differential_weights_new, d_differential_weights_old, self_adapting_differential_weight_probability, population_size);
                HIP_ASSERT(hipDeviceSynchronize());
            #endif
            #ifdef USE_CUDA
                cuda_wrapper::gpu_set_crossover_probabilities_new_wrapper(
                        dim3(grid_size_self_adapting_parameters), dim3(GPU_BLOCK_SIZE), stream_array[0],
                        d_rng_state, d_crossover_probabilities_new, d_crossover_probabilities_old, self_adapting_crossover_probability, population_size);
                cuda_wrapper::gpu_set_differential_weights_new_wrapper(
                        dim3(grid_size_self_adapting_parameters), dim3(GPU_BLOCK_SIZE), stream_array[1 % MAX_GPU_STREAMS],
                        d_rng_state + 4*population_size, d_differential_weights_new, d_differential_weights_old, self_adapting_differential_weight_probability, population_size);
                CUDA_ASSERT(cudaDeviceSynchronize());
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
            //Set crossover probabilities and differential weights
            for (int i=0; i<population_size; i++) {
                if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
                    crossover_probabilities_new[i] = (xoshiro256p(rng) >> 11) * 0x1.0p-53;
                } else {
                    crossover_probabilities_new[i] = crossover_probabilities_old[i];
                }

                if ((xoshiro256p(rng) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
                    //differential_weights_new[i] = 
                    //    self_adapting_differential_weight_shift + 
                    //    self_adapting_differential_weight*((xoshiro256p(rng) >> 11) * 0x1.0p-53);
                    differential_weights_new[i] = 2.0*((xoshiro256p(rng) >> 11) * 0x1.0p-53);
                } else {
                    differential_weights_new[i] = differential_weights_old[i];
                }
            }
        #endif

        #ifdef GPU_BLOCK_SIZE
            int grid_size_set_mutant_indices = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            int grid_size_set_mutate_indices = (population_size*genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            #ifndef USE_CUDA
                hipLaunchKernelGGL(gpu_set_mutant_indices,
                        dim3(grid_size_set_mutant_indices), dim3(GPU_BLOCK_SIZE), 0, stream_array[0],
                        d_rng_state, d_mutant_indices, population_size);
                hipLaunchKernelGGL(gpu_set_mutate_indices,
                        dim3(grid_size_set_mutate_indices), dim3(GPU_BLOCK_SIZE), 0, stream_array[1 % MAX_GPU_STREAMS],
                        d_rng_state + 4*population_size, d_mutate_indices, d_crossover_probabilities_new, population_size, genome_size);
                HIP_ASSERT(hipDeviceSynchronize());
            #endif
            #ifdef USE_CUDA
                cuda_wrapper::gpu_set_mutant_indices_wrapper(
                        dim3(grid_size_set_mutant_indices), dim3(GPU_BLOCK_SIZE), stream_array[0],
                        d_rng_state, d_mutant_indices, population_size);
                cuda_wrapper::gpu_set_mutate_indices_wrapper(
                        dim3(grid_size_set_mutate_indices), dim3(GPU_BLOCK_SIZE), stream_array[1 % MAX_GPU_STREAMS],
                        d_rng_state + 4*population_size, d_mutate_indices, d_crossover_probabilities_new, population_size, genome_size);
                CUDA_ASSERT(cudaDeviceSynchronize());
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
        //Set mutant population and indices 
            for (int i=0; i<population_size; i++) {
                set_mutant_indices(rng, mutant_indices + 3*i, i, population_size);
                double crossover_rate = crossover_probabilities_new[i];
                for (int j=0; j<genome_size; j++) {
                    mutate_indices[i*genome_size + j] = (xoshiro256p(rng) >> 11) * 0x1.0p-53 < crossover_rate;
                }
            }
        #endif

        #ifdef GPU_BLOCK_SIZE
            int grid_size_set_population_new = (population_size*genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            #ifndef USE_CUDA
                hipLaunchKernelGGL(gpu_set_population_new,
                        dim3(grid_size_set_population_new), dim3(GPU_BLOCK_SIZE), 0, 0,
                        d_population_new, d_population_old, d_mutant_indices, d_differential_weights_new, d_mutate_indices, population_size, genome_size);
                HIP_ASSERT(hipDeviceSynchronize());
            #endif
            #ifdef USE_CUDA
                cuda_wrapper::gpu_set_population_new_wrapper(
                        dim3(grid_size_set_population_new), dim3(GPU_BLOCK_SIZE),
                        d_population_new, d_population_old, d_mutant_indices, d_differential_weights_new, d_mutate_indices, population_size, genome_size);
                CUDA_ASSERT(cudaDeviceSynchronize());
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
            for (int i=0; i<population_size; i++) {
                double F = differential_weights_new[i];
                int mutant_index1 = mutant_indices[3*i];
                int mutant_index2 = mutant_indices[3*i + 1];
                int mutant_index3 = mutant_indices[3*i + 2];
                for (int j=0; j<genome_size; j++) {
                    bool mutate = mutate_indices[i*genome_size + j];
                    if (mutate) {
                        population_new[i*genome_size + j] = fabs( 
                            population_old[mutant_index1*genome_size + j] + F*(
                                    population_old[mutant_index2*genome_size + j] -
                                    population_old[mutant_index3*genome_size + j]));
                    } else {
                        population_new[i*genome_size + j] = population_old[i*genome_size + j];
                    }
                }
            }
        #endif

        // Normalization
        if (normalize) {
            #ifdef GPU_BLOCK_SIZE
                int grid_size_set_normalization = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                #ifndef USE_CUDA
                    HIP_ASSERT(hipMemset(d_normalization,0, bytes_normalization));
                    for (int i=0; i<population_size; i++) {
                        int stream_idx = i % MAX_GPU_STREAMS;
                        hipLaunchKernelGGL(gpu_matrix_multiply_MxN_by_Nx1,
                                dim3(grid_size_set_normalization), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                                d_normalization, d_population_new, d_normalization_term, genome_size, i);
                    }
                    HIP_ASSERT(hipDeviceSynchronize());
                    int grid_size_normalize_population = (population_size*genome_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
                    hipLaunchKernelGGL(gpu_normalize_population, dim3(grid_size_normalize_population), dim3(GPU_BLOCK_SIZE), 0, 0,
                            d_population_new, d_normalization, zeroth_moment, population_size, genome_size); 
                    HIP_ASSERT(hipDeviceSynchronize());
                #endif
                #ifdef USE_CUDA
                    CUDA_ASSERT(cudaMemset(d_normalization,0, bytes_normalization));
                    for (int i=0; i<population_size; i++) {
                        int stream_idx = i % MAX_GPU_STREAMS;
                        cuda_wrapper::gpu_matrix_multiply_MxN_by_Nx1_wrapper(
                                dim3(grid_size_set_normalization), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                                d_normalization, d_population_new, d_normalization_term, genome_size, i);
                    }
                    CUDA_ASSERT(cudaDeviceSynchronize());
                    int grid_size_normalize_population = (population_size*genome_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
                    cuda_wrapper::gpu_normalize_population_wrapper( dim3(grid_size_normalize_population), dim3(GPU_BLOCK_SIZE),
                            d_population_new, d_normalization, zeroth_moment, population_size, genome_size); 
                    CUDA_ASSERT(cudaDeviceSynchronize());
                #endif
            #endif
            #ifndef GPU_BLOCK_SIZE
                matrix_multiply_MxN_by_Nx1(normalization, population_new,
                        normalization_term, population_size, genome_size);
                for (int i=0; i<population_size; i++) {
                    double _norm = normalization[i];
                    for (int j=0; j<genome_size; j++) {
                        population_new[i*genome_size + j] *= zeroth_moment/_norm;
                    }
                }
            #endif
        }

        //Rejection
        //Set model isf for new population
        #ifdef GPU_BLOCK_SIZE
            int grid_size_set_isf_model = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            #ifndef USE_CUDA
                HIP_ASSERT(hipMemset(d_isf_model,0, bytes_isf_model));
                for (int i=0; i<population_size*number_of_timeslices; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    hipLaunchKernelGGL(gpu_matrix_multiply_LxM_by_MxN,
                            dim3(grid_size_set_isf_model), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                            d_isf_model, d_isf_term, d_population_new, number_of_timeslices, genome_size, i);
                }
                HIP_ASSERT(hipDeviceSynchronize());
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMemset(d_isf_model,0, bytes_isf_model));
                for (int i=0; i<population_size*number_of_timeslices; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    cuda_wrapper::gpu_matrix_multiply_LxM_by_MxN_wrapper(
                            dim3(grid_size_set_isf_model), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                            d_isf_model, d_isf_term, d_population_new, number_of_timeslices, genome_size, i);
                }
                CUDA_ASSERT(cudaDeviceSynchronize());
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
            matrix_multiply_LxM_by_MxN(isf_model, isf_term, population_new,
                    number_of_timeslices, genome_size, population_size);
        #endif

        //Set moments
        if (use_inverse_first_moment) {
            #ifdef GPU_BLOCK_SIZE
                int grid_size_set_inverse_first_moments = (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                #ifndef USE_CUDA
                    HIP_ASSERT(hipMemset(d_inverse_first_moments,0, bytes_inverse_first_moments));
                    for (int i=0; i<population_size; i++) {
                        int stream_idx = i % MAX_GPU_STREAMS;
                        hipLaunchKernelGGL(gpu_matrix_multiply_MxN_by_Nx1,
                                dim3(grid_size_set_inverse_first_moments), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                                d_inverse_first_moments, d_isf_model, d_inverse_first_moments_term, number_of_timeslices, i);
                    }
                    HIP_ASSERT(hipDeviceSynchronize());
                #endif
                #ifdef USE_CUDA
                    CUDA_ASSERT(cudaMemset(d_inverse_first_moments,0, bytes_inverse_first_moments));
                    for (int i=0; i<population_size; i++) {
                        int stream_idx = i % MAX_GPU_STREAMS;
                        cuda_wrapper::gpu_matrix_multiply_MxN_by_Nx1_wrapper(
                                dim3(grid_size_set_inverse_first_moments), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                                d_inverse_first_moments, d_isf_model, d_inverse_first_moments_term, number_of_timeslices, i);
                    }
                    CUDA_ASSERT(cudaDeviceSynchronize());
                #endif
            #endif
            #ifndef GPU_BLOCK_SIZE
                matrix_multiply_MxN_by_Nx1(inverse_first_moments, isf_model,
                        inverse_first_moments_term, population_size,
                        number_of_timeslices);
            #endif
        }
        if (use_first_moment) {
            #ifdef GPU_BLOCK_SIZE
                int grid_size_set_first_moments = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                #ifndef USE_CUDA
                    HIP_ASSERT(hipMemset(d_first_moments,0, bytes_first_moments));
                    for (int i=0; i<population_size; i++) {
                        int stream_idx = i % MAX_GPU_STREAMS;
                        hipLaunchKernelGGL(gpu_matrix_multiply_MxN_by_Nx1,
                                dim3(grid_size_set_first_moments), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                                d_first_moments, d_population_new, d_first_moments_term, genome_size, i);
                    }
                    HIP_ASSERT(hipDeviceSynchronize());
                #endif
                #ifdef USE_CUDA
                    CUDA_ASSERT(cudaMemset(d_first_moments,0, bytes_first_moments));
                    for (int i=0; i<population_size; i++) {
                        int stream_idx = i % MAX_GPU_STREAMS;
                        cuda_wrapper::gpu_matrix_multiply_MxN_by_Nx1_wrapper(
                                dim3(grid_size_set_first_moments), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                                d_first_moments, d_population_new, d_first_moments_term, genome_size, i);
                    }
                    CUDA_ASSERT(cudaDeviceSynchronize());
                #endif
            #endif
            #ifndef GPU_BLOCK_SIZE
                matrix_multiply_MxN_by_Nx1(first_moments, population_new,
                        first_moments_term, population_size, genome_size);
            #endif
        }
        if (use_third_moment) {
            #ifdef GPU_BLOCK_SIZE
                int grid_size_set_third_moments = (genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                #ifndef USE_CUDA
                    HIP_ASSERT(hipMemset(d_third_moments,0, bytes_third_moments));
                    for (int i=0; i<population_size; i++) {
                        int stream_idx = i % MAX_GPU_STREAMS;
                        hipLaunchKernelGGL(gpu_matrix_multiply_MxN_by_Nx1,
                                dim3(grid_size_set_third_moments), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                                d_third_moments, d_population_new, d_third_moments_term, genome_size, i);
                    }
                    HIP_ASSERT(hipDeviceSynchronize());
                #endif
                #ifdef USE_CUDA
                    CUDA_ASSERT(cudaMemset(d_third_moments,0, bytes_third_moments));
                    for (int i=0; i<population_size; i++) {
                        int stream_idx = i % MAX_GPU_STREAMS;
                        cuda_wrapper::gpu_matrix_multiply_MxN_by_Nx1_wrapper(
                                dim3(grid_size_set_third_moments), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                                d_third_moments, d_population_new, d_third_moments_term, genome_size, i);
                    }
                    CUDA_ASSERT(cudaDeviceSynchronize());
                #endif
            #endif
            #ifndef GPU_BLOCK_SIZE
                matrix_multiply_MxN_by_Nx1(third_moments, population_new,
                        third_moments_term, population_size, genome_size);
            #endif
        }

        //Set fitness for new population
        #ifdef GPU_BLOCK_SIZE
            int grid_size_set_fitness = (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            int grid_size_set_fitness_moments = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
            #ifndef USE_CUDA
                HIP_ASSERT(hipMemset(d_fitness_new,0, bytes_fitness_new));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    hipLaunchKernelGGL(gpu_set_fitness,
                            dim3(grid_size_set_fitness), dim3(GPU_BLOCK_SIZE), 0, stream_array[stream_idx],
                            d_fitness_new, d_isf, d_isf_model, d_isf_error, number_of_timeslices, i);
                }
                HIP_ASSERT(hipDeviceSynchronize());
                if (use_inverse_first_moment) {
                    hipLaunchKernelGGL(gpu_set_fitness_moments_reduced_chi_squared,
                            dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE), 0, 0,
                            d_fitness_new, d_inverse_first_moments, inverse_first_moment, inverse_first_moment_error, population_size);
                    HIP_ASSERT(hipDeviceSynchronize());
                }
                if (use_first_moment) {
                    hipLaunchKernelGGL(gpu_set_fitness_moments_chi_squared,
                            dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE), 0, 0,
                            d_fitness_new, d_first_moments, first_moment, population_size);
                    HIP_ASSERT(hipDeviceSynchronize());
                }
                if (use_third_moment) {
                    hipLaunchKernelGGL(gpu_set_fitness_moments_reduced_chi_squared,
                            dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE), 0, 0,
                            d_fitness_new, d_third_moments, third_moment, third_moment_error, population_size);
                    HIP_ASSERT(hipDeviceSynchronize());
                }

                //Rejection step
                int grid_size_set_rejection_indices = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                int grid_size_swap_control_parameters = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                int grid_size_swap_populations = (population_size*genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                hipLaunchKernelGGL(gpu_set_rejection_indices,
                        dim3(grid_size_set_rejection_indices), dim3(GPU_BLOCK_SIZE), 0, 0,
                        d_rejection_indices, d_fitness_new, d_fitness_old, population_size);
                HIP_ASSERT(hipDeviceSynchronize());
                hipLaunchKernelGGL(gpu_swap_control_parameters,
                        dim3(grid_size_swap_control_parameters), dim3(GPU_BLOCK_SIZE), 0, stream_array[0],
                        d_crossover_probabilities_old, d_crossover_probabilities_new, d_rejection_indices, population_size);
                hipLaunchKernelGGL(gpu_swap_control_parameters,
                        dim3(grid_size_swap_control_parameters), dim3(GPU_BLOCK_SIZE), 0, stream_array[1 % MAX_GPU_STREAMS],
                        d_differential_weights_old, d_differential_weights_new, d_rejection_indices, population_size);
                hipLaunchKernelGGL(gpu_swap_populations,
                        dim3(grid_size_swap_populations), dim3(GPU_BLOCK_SIZE), 0, stream_array[2 % MAX_GPU_STREAMS],
                        d_population_old, d_population_new, d_rejection_indices, population_size, genome_size);
                HIP_ASSERT(hipDeviceSynchronize());
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaMemset(d_fitness_new,0, bytes_fitness_new));
                for (int i=0; i<population_size; i++) {
                    int stream_idx = i % MAX_GPU_STREAMS;
                    cuda_wrapper::gpu_set_fitness_wrapper(
                            dim3(grid_size_set_fitness), dim3(GPU_BLOCK_SIZE), stream_array[stream_idx],
                            d_fitness_new, d_isf, d_isf_model, d_isf_error, number_of_timeslices, i);
                }
                CUDA_ASSERT(cudaDeviceSynchronize());
                if (use_inverse_first_moment) {
                    cuda_wrapper::gpu_set_fitness_moments_reduced_chi_squared_wrapper(
                            dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE),
                            d_fitness_new, d_inverse_first_moments, inverse_first_moment, inverse_first_moment_error, population_size);
                    CUDA_ASSERT(cudaDeviceSynchronize());
                }
                if (use_first_moment) {
                    cuda_wrapper::gpu_set_fitness_moments_chi_squared_wrapper(
                            dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE),
                            d_fitness_new, d_first_moments, first_moment, population_size);
                    CUDA_ASSERT(cudaDeviceSynchronize());
                }
                if (use_third_moment) {
                    cuda_wrapper::gpu_set_fitness_moments_reduced_chi_squared_wrapper(
                            dim3(grid_size_set_fitness_moments), dim3(GPU_BLOCK_SIZE),
                            d_fitness_new, d_third_moments, third_moment, third_moment_error, population_size);
                    CUDA_ASSERT(cudaDeviceSynchronize());
                }

                //Rejection step
                int grid_size_set_rejection_indices = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                int grid_size_swap_control_parameters = (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                int grid_size_swap_populations = (population_size*genome_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
                cuda_wrapper::gpu_set_rejection_indices_wrapper(
                        dim3(grid_size_set_rejection_indices), dim3(GPU_BLOCK_SIZE),
                        d_rejection_indices, d_fitness_new, d_fitness_old, population_size);
                CUDA_ASSERT(cudaDeviceSynchronize());
                cuda_wrapper::gpu_swap_control_parameters_wrapper(
                        dim3(grid_size_swap_control_parameters), dim3(GPU_BLOCK_SIZE), stream_array[0],
                        d_crossover_probabilities_old, d_crossover_probabilities_new, d_rejection_indices, population_size);
                cuda_wrapper::gpu_swap_control_parameters_wrapper(
                        dim3(grid_size_swap_control_parameters), dim3(GPU_BLOCK_SIZE), stream_array[1 % MAX_GPU_STREAMS],
                        d_differential_weights_old, d_differential_weights_new, d_rejection_indices, population_size);
                cuda_wrapper::gpu_swap_populations_wrapper(
                        dim3(grid_size_swap_populations), dim3(GPU_BLOCK_SIZE), stream_array[2 % MAX_GPU_STREAMS],
                        d_population_old, d_population_new, d_rejection_indices, population_size, genome_size);
                CUDA_ASSERT(cudaDeviceSynchronize());
            #endif
        #endif
        #ifndef GPU_BLOCK_SIZE
            for (int i=0; i<population_size; i++) {
                double _fitness = reduced_chi_square_statistic(isf,
                        isf_model + i*number_of_timeslices, isf_error,
                        number_of_timeslices)/number_of_timeslices;
                if (use_inverse_first_moment) {
                    _fitness += pow((inverse_first_moment - inverse_first_moments[i])/inverse_first_moment_error,2);
                }
                if (use_first_moment) {
                    _fitness += pow(first_moments[i] - first_moment,2)/first_moment;
                }
                if (use_third_moment) {
                    _fitness += pow((third_moment - third_moments[i])/third_moment_error,2);
                }
                // Rejection step
                if (_fitness <= fitness_old[i]) {
                    fitness_old[i] = _fitness;
                    crossover_probabilities_old[i] = crossover_probabilities_new[i];
                    differential_weights_old[i] = differential_weights_new[i];
                    for (int j=0; j<genome_size; j++) {
                        population_old[i*genome_size + j] = population_new[i*genome_size + j];
                    }
                }
            }
        #endif
    }

    //Transfer data from gpu to host
    #ifdef GPU_BLOCK_SIZE
        int grid_size_set_fitness_standard_deviation_sqrt = (generation + 1 + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE;
        #ifndef USE_CUDA
            HIP_ASSERT(hipDeviceSynchronize());
            HIP_ASSERT(hipMemcpy(fitness_old, d_fitness_old, bytes_fitness_old, hipMemcpyDeviceToHost));
            HIP_ASSERT(hipMemcpy(population_old, d_population_old, bytes_population, hipMemcpyDeviceToHost));
            if (track_stats) {
                hipLaunchKernelGGL(gpu_set_fitness_standard_deviation_sqrt,
                        dim3(grid_size_set_fitness_standard_deviation_sqrt), dim3(GPU_BLOCK_SIZE), 0, 0,
                        d_fitness_standard_deviation, generation + 1);
                HIP_ASSERT(hipDeviceSynchronize());
                HIP_ASSERT(hipMemcpy(fitness_mean, d_fitness_mean, bytes_fitness_mean, hipMemcpyDeviceToHost));
                HIP_ASSERT(hipMemcpy(fitness_minimum, d_fitness_minimum, bytes_fitness_minimum, hipMemcpyDeviceToHost));
                HIP_ASSERT(hipMemcpy(fitness_standard_deviation, d_fitness_standard_deviation, bytes_fitness_standard_deviation, hipMemcpyDeviceToHost));
            }
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaDeviceSynchronize());
            CUDA_ASSERT(cudaMemcpy(fitness_old, d_fitness_old, bytes_fitness_old, cudaMemcpyDeviceToHost));
            CUDA_ASSERT(cudaMemcpy(population_old, d_population_old, bytes_population, cudaMemcpyDeviceToHost));
            if (track_stats) {
                cuda_wrapper::gpu_set_fitness_standard_deviation_sqrt_wrapper(
                        dim3(grid_size_set_fitness_standard_deviation_sqrt), dim3(GPU_BLOCK_SIZE),
                        d_fitness_standard_deviation, generation + 1);
                CUDA_ASSERT(cudaDeviceSynchronize());
                CUDA_ASSERT(cudaMemcpy(fitness_mean, d_fitness_mean, bytes_fitness_mean, cudaMemcpyDeviceToHost));
                CUDA_ASSERT(cudaMemcpy(fitness_minimum, d_fitness_minimum, bytes_fitness_minimum, cudaMemcpyDeviceToHost));
                CUDA_ASSERT(cudaMemcpy(fitness_standard_deviation, d_fitness_standard_deviation, bytes_fitness_standard_deviation, cudaMemcpyDeviceToHost));
            }
        #endif
    #endif

    std::tie(minimum_fitness_idx,minimum_fitness) = argmin_and_min(fitness_old,population_size);

    double * best_dsf;
    best_dsf = (double*) malloc(sizeof(double)*genome_size);
    for (int i=0; i<genome_size; i++) {
        double f = frequency[i];
        best_dsf[i] = 0.5*population_old[genome_size*minimum_fitness_idx + i]*exp(0.5*beta*f);
    }

    //Get Statistics
    if (generation == number_of_generations - 2) {
        generation += 1;
        if (track_stats) {
            fitness_mean[generation] = mean(fitness_old, population_size);
            fitness_minimum[generation] = minimum_fitness;
            fitness_standard_deviation[generation] = standard_deviation(fitness_old,
                    fitness_mean[generation], population_size);
        }
    }

    //Save data
    std::string best_dsf_filename_str = string_format("deac_dsf_%s.bin",uuid_str.c_str());
    fs::path best_dsf_filename = save_directory / best_dsf_filename_str;
    write_array(best_dsf_filename, best_dsf, genome_size);
    std::string frequency_filename_str = string_format("deac_frequency_%s.bin",uuid_str.c_str());
    fs::path frequency_filename = save_directory / frequency_filename_str;
    write_array(frequency_filename, frequency, genome_size);
    fs::path fitness_mean_filename;
    fs::path fitness_minimum_filename;
    fs::path fitness_standard_deviation_filename;
    if (track_stats) {
        std::string fitness_mean_filename_str = string_format("deac_stats_fitness-mean_%s.bin",uuid_str.c_str());
        std::string fitness_minimum_filename_str = string_format("deac_stats_fitness-minimum_%s.bin",uuid_str.c_str());
        std::string fitness_standard_deviation_filename_str = string_format("deac_stats_fitness-standard-deviation_%s.bin",uuid_str.c_str());
        fs::path fitness_mean_filename = save_directory / fitness_mean_filename_str;
        fs::path fitness_minimum_filename = save_directory / fitness_minimum_filename_str;
        fs::path fitness_standard_deviation_filename = save_directory / fitness_standard_deviation_filename_str;
        write_array(fitness_mean_filename, fitness_mean, generation + 1);
        write_array(fitness_minimum_filename, fitness_minimum, generation + 1);
        write_array(fitness_standard_deviation_filename, fitness_standard_deviation, generation + 1);
    }

    //Write to log file
    std::string log_filename_str = string_format("deac_log_%s.dat",uuid_str.c_str());
    fs::path log_filename = save_directory / log_filename_str;
    std::ofstream log_ofs(log_filename.c_str(), std::ios_base::out | std::ios_base::app );
    log_ofs << "uuid: " << uuid_str << std::endl;

    //Input parameters
    log_ofs << "temperature: " << temperature << std::endl;
    log_ofs << "number_of_generations: " << number_of_generations << std::endl;
    log_ofs << "number_of_timeslices: " << number_of_timeslices << std::endl;
    log_ofs << "population_size: " << population_size << std::endl;
    log_ofs << "genome_size: " << genome_size << std::endl;
    log_ofs << "normalize: " << normalize << std::endl;
    log_ofs << "use_inverse_first_moment: " << use_inverse_first_moment << std::endl;
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
    log_ofs << "minimum_fitness: " << minimum_fitness << std::endl;
    if (track_stats) {
        log_ofs << "fitness_mean_filename: " << fitness_mean_filename << std::endl;
        log_ofs << "fitness_minimum_filename: " << fitness_minimum_filename << std::endl;
        log_ofs << "fitness_standard_deviation_filename: " << fitness_standard_deviation_filename << std::endl;
    }
    log_ofs.close();

    //Free memory
    free(isf_term);
    free(population_old);
    free(population_new);
    free(fitness_old);
    if (normalize) {
        free(normalization_term);
        free(normalization);
    }
    if (use_first_moment) {
        free(first_moments_term);
        free(first_moments);
    }
    if (use_third_moment) {
        free(third_moments_term);
        free(third_moments);
    }
    free(isf_model);
    if (use_inverse_first_moment) {
        free(inverse_first_moments_term);
        free(inverse_first_moments);
    }
    free(crossover_probabilities_old);
    free(crossover_probabilities_new);
    free(differential_weights_old);
    free(differential_weights_new);
    if (track_stats) {
        free(fitness_mean);
        free(fitness_minimum);
        free(fitness_standard_deviation);
    }
    free(mutate_indices);
    free(mutant_indices);
    free(best_dsf);

    
    #ifdef GPU_BLOCK_SIZE
        free(rng_state);
        // Release device memory
        #ifndef USE_CUDA
            HIP_ASSERT(hipFree(d_isf));
            HIP_ASSERT(hipFree(d_isf_error));
            HIP_ASSERT(hipFree(d_isf_term));
            HIP_ASSERT(hipFree(d_population_old));
            HIP_ASSERT(hipFree(d_population_new));
            HIP_ASSERT(hipFree(d_fitness_old));
            HIP_ASSERT(hipFree(d_fitness_new));
            if (normalize) {
                HIP_ASSERT(hipFree(d_normalization_term));
                HIP_ASSERT(hipFree(d_normalization));
            }
            if (use_first_moment) {
                HIP_ASSERT(hipFree(d_first_moments_term));
                HIP_ASSERT(hipFree(d_first_moments));
            }
            if (use_third_moment) {
                HIP_ASSERT(hipFree(d_third_moments_term));
                HIP_ASSERT(hipFree(d_third_moments));
            }
            HIP_ASSERT(hipFree(d_isf_model));
            if (use_inverse_first_moment) {
                HIP_ASSERT(hipFree(d_inverse_first_moments_term));
                HIP_ASSERT(hipFree(d_inverse_first_moments));
            }
            HIP_ASSERT(hipFree(d_crossover_probabilities_old));
            HIP_ASSERT(hipFree(d_crossover_probabilities_new));
            HIP_ASSERT(hipFree(d_differential_weights_old));
            HIP_ASSERT(hipFree(d_differential_weights_new));
            if (track_stats) {
                HIP_ASSERT(hipFree(d_fitness_mean));
                HIP_ASSERT(hipFree(d_fitness_minimum));
                HIP_ASSERT(hipFree(d_fitness_standard_deviation));
            }
            HIP_ASSERT(hipFree(d_mutate_indices));
            HIP_ASSERT(hipFree(d_rejection_indices));
            HIP_ASSERT(hipFree(d_mutant_indices));
            HIP_ASSERT(hipFree(d_minimum_fitness));
            HIP_ASSERT(hipFree(d_rng_state));
        #endif
        #ifdef USE_CUDA
            CUDA_ASSERT(cudaFree(d_isf));
            CUDA_ASSERT(cudaFree(d_isf_error));
            CUDA_ASSERT(cudaFree(d_isf_term));
            CUDA_ASSERT(cudaFree(d_population_old));
            CUDA_ASSERT(cudaFree(d_population_new));
            CUDA_ASSERT(cudaFree(d_fitness_old));
            CUDA_ASSERT(cudaFree(d_fitness_new));
            if (normalize) {
                CUDA_ASSERT(cudaFree(d_normalization_term));
                CUDA_ASSERT(cudaFree(d_normalization));
            }
            if (use_first_moment) {
                CUDA_ASSERT(cudaFree(d_first_moments_term));
                CUDA_ASSERT(cudaFree(d_first_moments));
            }
            if (use_third_moment) {
                CUDA_ASSERT(cudaFree(d_third_moments_term));
                CUDA_ASSERT(cudaFree(d_third_moments));
            }
            CUDA_ASSERT(cudaFree(d_isf_model));
            if (use_inverse_first_moment) {
                CUDA_ASSERT(cudaFree(d_inverse_first_moments_term));
                CUDA_ASSERT(cudaFree(d_inverse_first_moments));
            }
            CUDA_ASSERT(cudaFree(d_crossover_probabilities_old));
            CUDA_ASSERT(cudaFree(d_crossover_probabilities_new));
            CUDA_ASSERT(cudaFree(d_differential_weights_old));
            CUDA_ASSERT(cudaFree(d_differential_weights_new));
            if (track_stats) {
                CUDA_ASSERT(cudaFree(d_fitness_mean));
                CUDA_ASSERT(cudaFree(d_fitness_minimum));
                CUDA_ASSERT(cudaFree(d_fitness_standard_deviation));
            }
            CUDA_ASSERT(cudaFree(d_mutate_indices));
            CUDA_ASSERT(cudaFree(d_rejection_indices));
            CUDA_ASSERT(cudaFree(d_mutant_indices));
            CUDA_ASSERT(cudaFree(d_minimum_fitness));
            CUDA_ASSERT(cudaFree(d_rng_state));
        #endif
        // Destroy Streams
        for (int i = 0; i < MAX_GPU_STREAMS; i++) {
            #ifndef USE_CUDA
                HIP_ASSERT(hipStreamDestroy(stream_array[i]));
            #endif
            #ifdef USE_CUDA
                CUDA_ASSERT(cudaStreamDestroy(stream_array[i]));
            #endif
        }
    #endif
}

int main (int argc, char *argv[]) {
    argparse::ArgumentParser program("DEAC");
    program.add_argument("-T", "--temperature")
        .help("Temperature of system.")
        .default_value(0.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("-N", "--number_of_generations")
        .help("Number of generations before genetic algorithm quits.")
        .default_value(100000)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("-P","--population_size")
        .help("Size of initial population")
        .default_value(512)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("-M","--genome_size")
        .help("Size of genome.")
        .default_value(512)
        .action([](const std::string& value) { return std::stoi(value); });
    program.add_argument("--omega_max")
        .help("Maximum frequency to explore.")
        .default_value(60.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--normalize")
        .help("Normalize spectrum to the zeroeth moment.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--use_inverse_first_moment")
        .help("Calculate inverse first moment from ISF data and use it in fitness.")
        .default_value(false)
        .implicit_value(true);
    program.add_argument("--first_moment")
        .help("FIXME First moment.")
        .default_value(-1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--third_moment")
        .help("FIXME Third moment.")
        .default_value(-1.0)
        .action([](const std::string& value) { return std::stod(value); });
    program.add_argument("--third_moment_error")
        .help("FIXME Third moment error.")
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
        .action([](const std::string& value) { return std::stoi(value); });
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
    program.add_argument("isf_file")
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

    unsigned int number_of_elements;
    double* numpy_data;
    std::string isf_file = program.get<std::string>("isf_file");
    std::tie(numpy_data,number_of_elements) = load_numpy_array(isf_file);
    int number_of_timeslices = number_of_elements/3;

    double * const imaginary_time = numpy_data;
    double * const isf = numpy_data + number_of_timeslices;
    double * const isf_error = numpy_data + 2*number_of_timeslices;

    uint64_t seed = 1407513600 + static_cast<uint64_t>(program.get<int>("--seed"));
    int seed_int = program.get<int>("--seed");
    struct xoshiro256p_state rng = xoshiro256p_init(seed);

    double temperature = program.get<double>("--temperature");
    int number_of_generations = program.get<int>("--number_of_generations");
    int population_size = program.get<int>("--population_size");
    int genome_size = program.get<int>("--genome_size");
    double max_frequency = program.get<double>("--omega_max");
    
    double * frequency;
    frequency = (double*) malloc(sizeof(double)*genome_size);
    double dfrequency = max_frequency/(genome_size - 1);
    for (int i=0; i<genome_size; i++) {
        frequency[i] = i*dfrequency;
    }

    bool normalize = program.get<bool>("--normalize");
    bool use_inverse_first_moment = program.get<bool>("--use_inverse_first_moment");
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
    std::string log_filename_str = string_format("deac_log_%s.dat",uuid_str.c_str());
    fs::path log_filename = save_directory / log_filename_str;
    std::ofstream log_ofs(log_filename.c_str(), std::ios_base::out | std::ios_base::app );

    //Input parameters
    log_ofs << "isf_file: " << isf_file << std::endl;
    log_ofs.close();

    deac( &rng, imaginary_time, isf, isf_error, frequency, temperature,
            number_of_generations, number_of_timeslices, population_size, genome_size,
            normalize, use_inverse_first_moment, first_moment, third_moment,
            third_moment_error, crossover_probability,
            self_adapting_crossover_probability, differential_weight,
            self_adapting_differential_weight_probability,
            self_adapting_differential_weight_shift,
            self_adapting_differential_weight, stop_minimum_fitness,
            track_stats, seed_int, uuid_str, save_directory);

    free(numpy_data);
    free(frequency);
    return 0;
}
