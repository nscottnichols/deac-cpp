#pragma once
#include "common_gpu.hpp"
#include "device_launch_parameters.h"
#include <stdint.h>

void gpu_dot(cudaStream_t s, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N);
void gpu_matmul(cudaStream_t s, int m, int n, int k, double alpha, double* __restrict__ A, double* __restrict__ B, double beta, double* __restrict__ C);
void gpu_get_minimum(cudaStream_t s, double* __restrict__ minimum, double* __restrict__ array, size_t N);
void gpu_normalize_population(cudaStream_t s, size_t grid_size, double* __restrict__ population, double* __restrict__ normalization, double zeroth_moment, size_t population_size, size_t genome_size);
void gpu_set_fitness(cudaStream_t s, double* __restrict__ fitness, double* __restrict__ isf, double* __restrict__ isf_model, double* __restrict__ isf_error, size_t number_of_timeslices);
void gpu_set_fitness_moments_reduced_chi_squared(cudaStream_t s, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, double moment_error, size_t population_size);
void gpu_set_fitness_moments_chi_squared(cudaStream_t s, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, size_t population_size);
void gpu_set_fitness_mean(cudaStream_t s, double* __restrict__ fitness_mean, double* __restrict__ fitness, size_t population_size);
void gpu_set_fitness_squared_mean(cudaStream_t s, double* __restrict__ fitness_squared_mean, double* __restrict__ fitness, size_t population_size);
void gpu_set_population_new(cudaStream_t s, size_t grid_size, double* __restrict__ population_new, double* __restrict__ population_old, size_t* __restrict__ mutant_indices, double* __restrict__ differential_weights_new, bool* __restrict__ mutate_indices, size_t population_size, size_t genome_size);
void gpu_match_population_zero(cudaStream_t s, size_t grid_size, double* __restrict__ population_negative_frequency, double* __restrict__ population_positive_frequency, size_t population_size, size_t genome_size);
void gpu_set_rejection_indices(cudaStream_t s, size_t grid_size, bool* __restrict__ rejection_indices, double* __restrict__ fitness_new, double* __restrict__ fitness_old, size_t population_size);
void gpu_swap_control_parameters(cudaStream_t s, size_t grid_size, double* __restrict__ control_parameter_old, double* __restrict__ control_parameter_new, bool* __restrict__ rejection_indices, size_t population_size);
void gpu_swap_populations(cudaStream_t s, size_t grid_size, double* __restrict__ population_old, double* __restrict__ population_new, bool* __restrict__ rejection_indices, size_t population_size, size_t genome_size);
void gpu_set_crossover_probabilities_new(cudaStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ crossover_probabilities_new, double* __restrict__ crossover_probabilities_old, double self_adapting_crossover_probability, size_t population_size);
void gpu_set_differential_weights_new(cudaStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ differential_weights_new, double* __restrict__ differential_weights_old, double self_adapting_differential_weight_probability, size_t population_size);
void gpu_set_mutant_indices(cudaStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, size_t* __restrict__ mutant_indices, size_t population_size);
void gpu_set_mutate_indices(cudaStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, bool* __restrict__ mutate_indices, double* __restrict__ crossover_probabilities, size_t population_size, size_t genome_size);

#ifdef USE_BLAS
    void gpu_blas_gemv(cublasHandle_t handle, int m, int n, double alpha, double* A, double* B, double beta, double* C);
    void gpu_blas_gemm(cublasHandle_t handle, int m, int n, int k, double alpha, double* A, double* B, double beta, double* C);
#endif
