#pragma once
#include "common_gpu.hpp"
#include "device_launch_parameters.h"

namespace cuda_wrapper {
#ifdef DEAC_DEBUG
    void gpu_check_array_wrapper(dim3, dim3, double *, int);
    void gpu_check_array_wrapper(dim3, dim3, cudaStream_t, double *, int);
#endif

    void gpu_matrix_multiply_MxN_by_Nx1_wrapper(dim3, dim3, double *, double *, double *, int, int);
    void gpu_matrix_multiply_MxN_by_Nx1_wrapper(dim3, dim3, cudaStream_t, double *, double *, double *, int, int);
    
    void gpu_matrix_multiply_LxM_by_MxN_wrapper(dim3, dim3, double *, double *, double *, int, int, int);
    void gpu_matrix_multiply_LxM_by_MxN_wrapper(dim3, dim3, cudaStream_t, double *, double *, double *, int, int, int);
    
    void gpu_normalize_population_wrapper(dim3, dim3, double *, double *, double, int, int);
    void gpu_normalize_population_wrapper(dim3, dim3, cudaStream_t, double *, double *, double, int, int);
    
    void gpu_set_fitness_wrapper(dim3, dim3, double *, double *, double *, double *, int, int);
    void gpu_set_fitness_wrapper(dim3, dim3, cudaStream_t, double *, double *, double *, double *, int, int);
    
    void gpu_set_fitness_moments_reduced_chi_squared_wrapper(dim3, dim3, double *, double *, double, double, int);
    void gpu_set_fitness_moments_reduced_chi_squared_wrapper(dim3, dim3, cudaStream_t, double *, double *, double, double, int);
    
    void gpu_set_fitness_moments_chi_squared_wrapper(dim3, dim3, double *, double *, double, int);
    void gpu_set_fitness_moments_chi_squared_wrapper(dim3, dim3, cudaStream_t, double *, double *, double, int);
    
    void gpu_get_minimum_fitness_wrapper(dim3, dim3, double *, double *, int);
    void gpu_get_minimum_fitness_wrapper(dim3, dim3, cudaStream_t, double *, double *, int);
    
    void gpu_set_fitness_mean_wrapper(dim3, dim3, double *, double *, int, int);
    void gpu_set_fitness_mean_wrapper(dim3, dim3, cudaStream_t, double *, double *, int, int);
    
    void gpu_set_fitness_standard_deviation_wrapper(dim3, dim3, double *, double *, double *, int, int);
    void gpu_set_fitness_standard_deviation_wrapper(dim3, dim3, cudaStream_t, double *, double *, double *, int, int);
    
    void gpu_set_fitness_standard_deviation_sqrt_wrapper(dim3, dim3, double *, int);
    void gpu_set_fitness_standard_deviation_sqrt_wrapper(dim3, dim3, cudaStream_t, double *, int);
    
    void gpu_set_population_new_wrapper(dim3, dim3, double *, double *, int *, double *, bool *, int, int);
    void gpu_set_population_new_wrapper(dim3, dim3, cudaStream_t, double *, double *, int *, double *, bool *, int, int);
    
    void gpu_set_rejection_indices_wrapper(dim3, dim3, bool *, double *, double *, int);
    void gpu_set_rejection_indices_wrapper(dim3, dim3, cudaStream_t, bool *, double *, double *, int);
    
    void gpu_swap_control_parameters_wrapper(dim3, dim3, double *, double *, bool *, int);
    void gpu_swap_control_parameters_wrapper(dim3, dim3, cudaStream_t, double *, double *, bool *, int);
    
    void gpu_swap_populations_wrapper(dim3, dim3, double *, double *, bool *, int, int);
    void gpu_swap_populations_wrapper(dim3, dim3, cudaStream_t, double *, double *, bool *, int, int);
}
