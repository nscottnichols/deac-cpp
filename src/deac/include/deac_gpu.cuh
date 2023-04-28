#pragma once
#include "common_gpu.hpp"
#include "device_launch_parameters.h"
#include <stdint.h>

namespace cuda_wrapper {
#ifdef DEAC_DEBUG
    void gpu_check_array_wrapper(dim3, dim3, double *, size_t);
    void gpu_check_array_wrapper(dim3, dim3, cudaStream_t, double *, size_t);
#endif

    void gpu_matrix_multiply_MxN_by_Nx1_wrapper(dim3, dim3, double *, double *, double *, size_t, size_t);
    void gpu_matrix_multiply_MxN_by_Nx1_wrapper(dim3, dim3, cudaStream_t, double *, double *, double *, size_t, size_t);
    
    void gpu_matrix_multiply_LxM_by_MxN_wrapper(dim3, dim3, double *, double *, double *, size_t, size_t, size_t);
    void gpu_matrix_multiply_LxM_by_MxN_wrapper(dim3, dim3, cudaStream_t, double *, double *, double *, size_t, size_t, size_t);
    
    void gpu_normalize_population_wrapper(dim3, dim3, double *, double *, double, size_t, size_t);
    void gpu_normalize_population_wrapper(dim3, dim3, cudaStream_t, double *, double *, double, size_t, size_t);
    
    void gpu_set_fitness_wrapper(dim3, dim3, double *, double *, double *, double *, size_t, size_t);
    void gpu_set_fitness_wrapper(dim3, dim3, cudaStream_t, double *, double *, double *, double *, size_t, size_t);
    
    void gpu_set_fitness_moments_reduced_chi_squared_wrapper(dim3, dim3, double *, double *, double, double, size_t);
    void gpu_set_fitness_moments_reduced_chi_squared_wrapper(dim3, dim3, cudaStream_t, double *, double *, double, double, size_t);
    
    void gpu_set_fitness_moments_chi_squared_wrapper(dim3, dim3, double *, double *, double, size_t);
    void gpu_set_fitness_moments_chi_squared_wrapper(dim3, dim3, cudaStream_t, double *, double *, double, size_t);
    
    void gpu_get_minimum_fitness_wrapper(dim3, dim3, double *, double *, size_t);
    void gpu_get_minimum_fitness_wrapper(dim3, dim3, cudaStream_t, double *, double *, size_t);
    
    void gpu_set_fitness_mean_wrapper(dim3, dim3, double *, double *, size_t, size_t);
    void gpu_set_fitness_mean_wrapper(dim3, dim3, cudaStream_t, double *, double *, size_t, size_t);
    
    void gpu_set_fitness_squared_mean_wrapper(dim3, dim3, double *, double *, size_t, size_t);
    void gpu_set_fitness_squared_mean_wrapper(dim3, dim3, cudaStream_t, double *, double *, size_t, size_t);
    
    void gpu_set_population_new_wrapper(dim3, dim3, double *, double *, size_t *, double *, bool *, size_t, size_t);
    void gpu_set_population_new_wrapper(dim3, dim3, cudaStream_t, double *, double *, size_t *, double *, bool *, size_t, size_t);
    
    void gpu_match_population_zero_wrapper(dim3, dim3, double *, double *, size_t, size_t);
    void gpu_match_population_zero_wrapper(dim3, dim3, cudaStream_t, double *, double *, size_t, size_t);
    
    void gpu_set_rejection_indices_wrapper(dim3, dim3, bool *, double *, double *, size_t);
    void gpu_set_rejection_indices_wrapper(dim3, dim3, cudaStream_t, bool *, double *, double *, size_t);
    
    void gpu_swap_control_parameters_wrapper(dim3, dim3, double *, double *, bool *, size_t);
    void gpu_swap_control_parameters_wrapper(dim3, dim3, cudaStream_t, double *, double *, bool *, size_t);
    
    void gpu_swap_populations_wrapper(dim3, dim3, double *, double *, bool *, size_t, size_t);
    void gpu_swap_populations_wrapper(dim3, dim3, cudaStream_t, double *, double *, bool *, size_t, size_t);

    void gpu_set_crossover_probabilities_new_wrapper(dim3, dim3, uint64_t *, double *, double *, double, size_t);
    void gpu_set_crossover_probabilities_new_wrapper(dim3, dim3, cudaStream_t, uint64_t *, double *, double *, double, size_t);
    
    void gpu_set_differential_weights_new_wrapper(dim3, dim3, uint64_t *, double *, double *, double, size_t);
    void gpu_set_differential_weights_new_wrapper(dim3, dim3, cudaStream_t, uint64_t *, double *, double *, double, size_t);
    
    void gpu_set_mutant_indices_wrapper(dim3, dim3, uint64_t *, size_t *, size_t);
    void gpu_set_mutant_indices_wrapper(dim3, dim3, cudaStream_t, uint64_t *, size_t *, size_t);
    
    void gpu_set_mutate_indices_wrapper(dim3, dim3, uint64_t *, bool *, double *, size_t, size_t);
    void gpu_set_mutate_indices_wrapper(dim3, dim3, cudaStream_t, uint64_t *, bool *, double *, size_t, size_t);

    void gpu_check_minimum_fitness_wrapper(dim3, dim3, double *, double);
    void gpu_check_minimum_fitness_wrapper(dim3, dim3, cudaStream_t, double *, double);
}
