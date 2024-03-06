/**
 * @file deac_gpu.cu
 * @author Nathan Nichols
 * @date 04.19.2021
 *
 * @brief GPU kernels using CUDA.
 */

#include "deac_gpu.cuh"

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNELS ---------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

__device__
uint64_t gpu_rol64(uint64_t x, uint64_t k) {
    return (x << k) | (x >> (64 - k));
}

__device__
uint64_t gpu_xoshiro256p_next(uint64_t * s) {
    uint64_t const result = s[0] + s[3];
    uint64_t const t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = gpu_rol64(s[3], 45);

    return result;
}

__device__
void sub_group_reduce_add(volatile double* _c, size_t local_idx) {
    #if (SUB_GROUP_SIZE >= 64)
        _c[local_idx] += _c[local_idx + 64];
    #endif
    #if (SUB_GROUP_SIZE >= 32)
        _c[local_idx] += _c[local_idx + 32];
    #endif
    #if (SUB_GROUP_SIZE >= 16)
        _c[local_idx] += _c[local_idx + 16];
    #endif
    #if (SUB_GROUP_SIZE >= 8)
        _c[local_idx] += _c[local_idx + 8];
    #endif
    #if (SUB_GROUP_SIZE >= 4)
        _c[local_idx] += _c[local_idx + 4];
    #endif
    #if (SUB_GROUP_SIZE >= 2)
        _c[local_idx] += _c[local_idx + 2];
    #endif
    #if (SUB_GROUP_SIZE >= 1)
        _c[local_idx] += _c[local_idx + 1];
    #endif
}

__device__
void gpu_reduce_add(double* _c) {
    size_t local_idx = threadIdx.x;
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        if (local_idx < 512) {
            _c[local_idx] += _c[local_idx + 512];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        if (local_idx < 256) {
            _c[local_idx] += _c[local_idx + 256];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        if (local_idx < 128) {
            _c[local_idx] += _c[local_idx + 128];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        if (local_idx < 64) {
            _c[local_idx] += _c[local_idx + 64];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
        if (local_idx < 32) {
            _c[local_idx] += _c[local_idx + 32];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
        if (local_idx < 16) {
            _c[local_idx] += _c[local_idx + 16];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
        if (local_idx < 8) {
            _c[local_idx] += _c[local_idx + 8];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
        if (local_idx < 4) {
            _c[local_idx] += _c[local_idx + 4];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
        if (local_idx < 2) {
            _c[local_idx] += _c[local_idx + 2];
        }
        __syncthreads();
    #endif

    //Sub-group reduce
    if (local_idx < SUB_GROUP_SIZE) {
        sub_group_reduce_add(_c, local_idx);
    }
    __syncthreads();
}

__device__
void sub_group_reduce_min(volatile double* _c, size_t local_idx) {
    #if (SUB_GROUP_SIZE >= 64)
        _c[local_idx] = _c[local_idx + 64] < _c[local_idx] ? _c[local_idx + 64] : _c[local_idx];
    #endif
    #if (SUB_GROUP_SIZE >= 32)
        _c[local_idx] = _c[local_idx + 32] < _c[local_idx] ? _c[local_idx + 32] : _c[local_idx];
    #endif
    #if (SUB_GROUP_SIZE >= 16)
        _c[local_idx] = _c[local_idx + 16] < _c[local_idx] ? _c[local_idx + 16] : _c[local_idx];
    #endif
    #if (SUB_GROUP_SIZE >= 8)
        _c[local_idx] = _c[local_idx + 8] < _c[local_idx] ? _c[local_idx + 8] : _c[local_idx];
    #endif
    #if (SUB_GROUP_SIZE >= 4)
        _c[local_idx] = _c[local_idx + 4] < _c[local_idx] ? _c[local_idx + 4] : _c[local_idx];
    #endif
    #if (SUB_GROUP_SIZE >= 2)
        _c[local_idx] = _c[local_idx + 2] < _c[local_idx] ? _c[local_idx + 2] : _c[local_idx];
    #endif
    #if (SUB_GROUP_SIZE >= 1)
        _c[local_idx] = _c[local_idx + 1] < _c[local_idx] ? _c[local_idx + 1] : _c[local_idx];
    #endif
}

__device__
void gpu_reduce_min(double* _c) {
    size_t local_idx = threadIdx.x;
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        if (local_idx < 512) {
            _c[local_idx] = _c[local_idx + 512] < _c[local_idx] ? _c[local_idx + 512] : _c[local_idx];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        if (local_idx < 256) {
            _c[local_idx] = _c[local_idx + 256] < _c[local_idx] ? _c[local_idx + 256] : _c[local_idx];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        if (local_idx < 128) {
            _c[local_idx] = _c[local_idx + 128] < _c[local_idx] ? _c[local_idx + 128] : _c[local_idx];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        if (local_idx < 64) {
            _c[local_idx] = _c[local_idx + 64] < _c[local_idx] ? _c[local_idx + 64] : _c[local_idx];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
        if (local_idx < 32) {
            _c[local_idx] = _c[local_idx + 32] < _c[local_idx] ? _c[local_idx + 32] : _c[local_idx];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
        if (local_idx < 16) {
            _c[local_idx] = _c[local_idx + 16] < _c[local_idx] ? _c[local_idx + 16] : _c[local_idx];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
        if (local_idx < 8) {
            _c[local_idx] = _c[local_idx + 8] < _c[local_idx] ? _c[local_idx + 8] : _c[local_idx];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
        if (local_idx < 4) {
            _c[local_idx] = _c[local_idx + 4] < _c[local_idx] ? _c[local_idx + 4] : _c[local_idx];
        }
        __syncthreads();
    #endif

    #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
        if (local_idx < 2) {
            _c[local_idx] = _c[local_idx + 2] < _c[local_idx] ? _c[local_idx + 2] : _c[local_idx];
        }
        __syncthreads();
    #endif

    //Sub-group reduce
    if (local_idx < SUB_GROUP_SIZE) {
        sub_group_reduce_min(_c, local_idx);
    }
    __syncthreads();
}

__global__
void gpu_dot(double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
    // C = B*A where [B] = 1xN and [A] = Nx1
    // Shared Local Memory _c
    __shared__ double _c[GPU_BLOCK_SIZE];
    // Set shared local memory _c
    size_t local_idx = threadIdx.x;
    if (local_idx < N) {
        _c[local_idx] = A[local_idx]*B[local_idx];
    } else {
        _c[local_idx] = 0.0;
    }

    for (size_t i = 1; i < (N + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
        size_t j = GPU_BLOCK_SIZE*i + local_idx;
        if (j < N) {
            _c[local_idx] += A[j]*B[j];
        }
    }
    __syncthreads();

    // Reduce _c (using shared local memory)
    gpu_reduce_add(_c);

    //Set C
    if (local_idx == 0) {
         C[0] = _c[0];
    }
}

__global__
void gpu_get_minimum(double* __restrict__ minimum, double* __restrict__ array, size_t N) {
    // finds minimum of array with length N
    // Shared Local Memory _c
    __shared__ double _c[GPU_BLOCK_SIZE];
    // Set shared local memory _c
    size_t local_idx = threadIdx.x;
    if (local_idx < N) {
        _c[local_idx] = array[local_idx];
    } else {
        _c[local_idx] = array[0];
    }

    for (size_t i = 1; i < (N + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
        size_t j = GPU_BLOCK_SIZE*i + local_idx;
        if (j < N) {
            _c[local_idx] = array[j] < _c[local_idx] ? array[j] : _c[local_idx];
        }
    }
    __syncthreads();

    // Reduce _c (using shared local memory)
    gpu_reduce_min(_c);

    //Set minimum
    if (local_idx == 0) {
         minimum[0] = _c[0];
    }
}

__global__
void gpu_normalize_population(double* __restrict__ population, double* __restrict__ normalization, double zeroth_moment, size_t population_size, size_t genome_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size*genome_size) {
        population[global_idx] *= zeroth_moment/normalization[global_idx/genome_size];
    }
}

__global__
void gpu_set_fitness(double* __restrict__ fitness, double* __restrict__ isf, double* __restrict__ isf_model, double* __restrict__ isf_error, size_t number_of_timeslices) {
    __shared__ double _f[GPU_BLOCK_SIZE];
    // Set shared local memory _f
    size_t local_idx = threadIdx.x;
    if (local_idx < number_of_timeslices) {
        double __f = (isf[local_idx] - isf_model[local_idx])/isf_error[local_idx];
        _f[local_idx] = __f*__f;
    } else {
        _f[local_idx] = 0.0;
    }

    for (size_t i = 1; i < (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
        size_t j = GPU_BLOCK_SIZE*i + local_idx;
        if (j < number_of_timeslices) {
            double __f = (isf[j] - isf_model[j])/isf_error[j];
            _f[local_idx] += __f*__f;
        }
    }
    __syncthreads();

    // Reduce _f (using shared local memory)
    gpu_reduce_add(_f);

    //Set fitness
    if (local_idx == 0) {
         fitness[0] += _f[0]/number_of_timeslices;
    }
}

__global__
void gpu_set_fitness_moments_reduced_chi_squared(double* __restrict__ fitness, double* __restrict__ moments, double moment, double moment_error, size_t population_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size) {
        double __f = (moment - moments[global_idx])/moment_error;
        fitness[global_idx] += __f*__f;
    }
}

__global__
void gpu_set_fitness_moments_chi_squared(double* __restrict__ fitness, double* __restrict__ moments, double moment, size_t population_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size) {
        double __f = moment - moments[global_idx];
        fitness[global_idx] += __f*__f;
    }
}

__global__
void gpu_set_fitness_mean(double* __restrict__ fitness_mean, double* __restrict__ fitness, size_t population_size) {
    __shared__ double _fm[GPU_BLOCK_SIZE];
    // Set shared local memory _fm
    size_t local_idx = threadIdx.x;
    if (local_idx < population_size) {
        _fm[local_idx] = fitness[local_idx];
    } else {
        _fm[local_idx] = 0.0;
    }

    for (size_t i = 1; i < (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
        size_t j = GPU_BLOCK_SIZE*i + local_idx;
        if (j < population_size) {
            _fm[local_idx] += fitness[j];
        }
    }
    __syncthreads();
    
    // Reduce _fm (using shared local memory)
    gpu_reduce_add(_fm);

    //Set fitness_mean
    if (local_idx == 0) {
         fitness_mean[0] += _fm[0]/population_size;
    }
}

__global__
void gpu_set_fitness_squared_mean(double* __restrict__ fitness_squared_mean, double* __restrict__ fitness, size_t population_size) {
    __shared__ double _fsm[GPU_BLOCK_SIZE];
    // Set shared local memory _fsm
    size_t local_idx = threadIdx.x;
    if (local_idx < population_size) {
        _fsm[local_idx] = fitness[local_idx]*fitness[local_idx];
    } else {
        _fsm[local_idx] = 0.0;
    }

    for (size_t i = 1; i < (population_size + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
        size_t j = GPU_BLOCK_SIZE*i + local_idx;
        if (j < population_size) {
            _fsm[local_idx] += fitness[j]*fitness[j];
        }
    }
    __syncthreads();
    
    // Reduce _fsm (using shared local memory)
    gpu_reduce_add(_fsm);

    //Set fitness_squared_mean
    if (local_idx == 0) {
         fitness_squared_mean[0] += _fsm[0]/population_size;
    }
}

__global__
void gpu_set_population_new(double* __restrict__ population_new, double* __restrict__ population_old, size_t* __restrict__ mutant_indices, double* __restrict__ differential_weights_new, bool* __restrict__ mutate_indices, size_t population_size, size_t genome_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size*genome_size) {
        size_t _i = global_idx/genome_size;
        size_t _j = global_idx - _i*genome_size;
        double F = differential_weights_new[_i];
        size_t mutant_index1 = mutant_indices[3*_i];
        size_t mutant_index2 = mutant_indices[3*_i + 1];
        size_t mutant_index3 = mutant_indices[3*_i + 2];
        bool mutate = mutate_indices[global_idx];
        if (mutate) {
            #ifdef ALLOW_NEGATIVE_SPECTRAL_WEIGHT
                population_new[global_idx] = population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]);
            #else
                population_new[global_idx] = fabs(population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]));
            #endif
        } else {
            population_new[global_idx] = population_old[global_idx];
        }
    }
}


__global__
void gpu_match_population_zero(double* __restrict__ population_negative_frequency, double* __restrict__ population_positive_frequency, size_t population_size, size_t genome_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size) {
        population_negative_frequency[global_idx*genome_size] = population_positive_frequency[global_idx*genome_size];
    }
}

__global__
void gpu_set_rejection_indices(bool* __restrict__ rejection_indices, double* __restrict__ fitness_new, double* __restrict__ fitness_old, size_t population_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size) {
        bool accept = fitness_new[global_idx] <= fitness_old[global_idx];
        rejection_indices[global_idx] = accept;
        if (accept) {
            fitness_old[global_idx] = fitness_new[global_idx];
        }
    }
}

__global__
void gpu_swap_control_parameters(double* __restrict__ control_parameter_old, double* __restrict__ control_parameter_new, bool* __restrict__ rejection_indices, size_t population_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size) {
        if (rejection_indices[global_idx]) {
            control_parameter_old[global_idx] = control_parameter_new[global_idx];
        }
    }
}

__global__
void gpu_swap_populations(double* __restrict__ population_old, double* __restrict__ population_new, bool* __restrict__ rejection_indices, size_t population_size, size_t genome_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size*genome_size) {
        size_t _i = global_idx/genome_size;
        if (rejection_indices[_i]) {
            population_old[global_idx] = population_new[global_idx];
        }
    }
}

__global__
void gpu_set_crossover_probabilities_new(uint64_t* __restrict__ rng_state, double* __restrict__ crossover_probabilities_new, double* __restrict__ crossover_probabilities_old, double self_adapting_crossover_probability, size_t population_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size) {
        if ((gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
            crossover_probabilities_new[global_idx] = (gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53;
        } else {
            crossover_probabilities_new[global_idx] = crossover_probabilities_old[global_idx];
        }
    }
}

__global__
void gpu_set_differential_weights_new(uint64_t* __restrict__ rng_state, double* __restrict__ differential_weights_new, double* __restrict__ differential_weights_old, double self_adapting_differential_weight_probability, size_t population_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size) {
        if ((gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
            differential_weights_new[global_idx] = 2.0*((gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53);
        } else {
            differential_weights_new[global_idx] = differential_weights_old[global_idx];
        }
    }
}

__device__
void gpu_set_mutant_indices(uint64_t* __restrict__ rng_state, size_t* __restrict__ mutant_indices, size_t mutant_index0, size_t length) {
    mutant_indices[0] = mutant_index0;
    mutant_indices[1] = mutant_index0;
    mutant_indices[2] = mutant_index0;
    while (mutant_indices[0] == mutant_index0) {
        mutant_indices[0] = gpu_xoshiro256p_next(rng_state) % length;
    }

    while ((mutant_indices[1] == mutant_index0) || (mutant_indices[1] == mutant_indices[0])) {
        mutant_indices[1] = gpu_xoshiro256p_next(rng_state) % length;
    }

    while ((mutant_indices[2] == mutant_index0) || (mutant_indices[2] == mutant_indices[0])
            || (mutant_indices[2] == mutant_indices[1])) {
        mutant_indices[2] = gpu_xoshiro256p_next(rng_state) % length;
    }
}

__global__
void gpu_set_mutant_indices(uint64_t* __restrict__ rng_state, size_t* __restrict__ mutant_indices, size_t population_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size) {
        gpu_set_mutant_indices(rng_state + 4*global_idx, mutant_indices + 3*global_idx, global_idx, population_size);
    }
}

__global__
void gpu_set_mutate_indices(uint64_t* __restrict__ rng_state, bool* __restrict__ mutate_indices, double* __restrict__ crossover_probabilities, size_t population_size, size_t genome_size) {
    size_t global_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (global_idx < population_size*genome_size) {
        size_t _i = global_idx/genome_size;
        mutate_indices[global_idx] = (gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < crossover_probabilities[_i];
    }
}

// Kernel Launcher
void gpu_dot(cudaStream_t s, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
    gpu_dot<<<dim3(1), dim3(GPU_BLOCK_SIZE), 0, s>>>(C, B, A, N);
}

void gpu_get_minimum(cudaStream_t s, double* __restrict__ minimum, double* __restrict__ array, size_t N) {
    gpu_get_minimum<<<dim3(1), dim3(GPU_BLOCK_SIZE), 0, s>>>(minimum, array, N);
}

void gpu_normalize_population(cudaStream_t s, size_t grid_size, double* __restrict__ population, double* __restrict__ normalization, double zeroth_moment, size_t population_size, size_t genome_size) {
    gpu_normalize_population<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(population, normalization, zeroth_moment, population_size, genome_size);
}

void gpu_set_fitness(cudaStream_t s, double* __restrict__ fitness, double* __restrict__ isf, double* __restrict__ isf_model, double* __restrict__ isf_error, size_t number_of_timeslices) {
    gpu_set_fitness<<<dim3(1), dim3(GPU_BLOCK_SIZE), 0, s>>>(fitness, isf, isf_model, isf_error, number_of_timeslices);
}

void gpu_set_fitness_moments_reduced_chi_squared(cudaStream_t s, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, double moment_error, size_t population_size) {
    gpu_set_fitness_moments_reduced_chi_squared<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(fitness, moments, moment, moment_error, population_size);
}

void gpu_set_fitness_moments_chi_squared(cudaStream_t s, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, size_t population_size) {
    gpu_set_fitness_moments_chi_squared<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(fitness, moments, moment, population_size);
}

void gpu_set_fitness_mean(cudaStream_t s, double* __restrict__ fitness_mean, double* __restrict__ fitness, size_t population_size) {
    gpu_set_fitness_mean<<<dim3(1), dim3(GPU_BLOCK_SIZE), 0, s>>>(fitness_mean, fitness, population_size);
}

void gpu_set_fitness_squared_mean(cudaStream_t s, double* __restrict__ fitness_squared_mean, double* __restrict__ fitness, size_t population_size) {
    gpu_set_fitness_squared_mean<<<dim3(1), dim3(GPU_BLOCK_SIZE), 0, s>>>(fitness_squared_mean, fitness, population_size);
}

void gpu_set_population_new(cudaStream_t s, size_t grid_size, double* __restrict__ population_new, double* __restrict__ population_old, size_t* __restrict__ mutant_indices, double* __restrict__ differential_weights_new, bool* __restrict__ mutate_indices, size_t population_size, size_t genome_size) {
    gpu_set_population_new<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(population_new, population_old, mutant_indices, differential_weights_new, mutate_indices, population_size, genome_size);
}

void gpu_match_population_zero(cudaStream_t s, size_t grid_size, double* __restrict__ population_negative_frequency, double* __restrict__ population_positive_frequency, size_t population_size, size_t genome_size) {
    gpu_match_population_zero<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(population_negative_frequency, population_positive_frequency, population_size, genome_size);
}

void gpu_set_rejection_indices(cudaStream_t s, size_t grid_size, bool* __restrict__ rejection_indices, double* __restrict__ fitness_new, double* __restrict__ fitness_old, size_t population_size) {
    gpu_set_rejection_indices<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(rejection_indices, fitness_new, fitness_old, population_size);
}

void gpu_swap_control_parameters(cudaStream_t s, size_t grid_size, double* __restrict__ control_parameter_old, double* __restrict__ control_parameter_new, bool* __restrict__ rejection_indices, size_t population_size) {
    gpu_swap_control_parameters<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(control_parameter_old, control_parameter_new, rejection_indices, population_size);
}

void gpu_swap_populations(cudaStream_t s, size_t grid_size, double* __restrict__ population_old, double* __restrict__ population_new, bool* __restrict__ rejection_indices, size_t population_size, size_t genome_size) {
    gpu_swap_populations<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(population_old, population_new, rejection_indices, population_size, genome_size);
}

void gpu_set_crossover_probabilities_new(cudaStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ crossover_probabilities_new, double* __restrict__ crossover_probabilities_old, double self_adapting_crossover_probability, size_t population_size) {
    gpu_set_crossover_probabilities_new<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(rng_state, crossover_probabilities_new, crossover_probabilities_old, self_adapting_crossover_probability, population_size);
}

void gpu_set_differential_weights_new(cudaStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ differential_weights_new, double* __restrict__ differential_weights_old, double self_adapting_differential_weight_probability, size_t population_size) {
    gpu_set_differential_weights_new<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(rng_state, differential_weights_new, differential_weights_old, self_adapting_differential_weight_probability, population_size);
}

void gpu_set_mutant_indices(cudaStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, size_t* __restrict__ mutant_indices, size_t population_size) {
    gpu_set_mutant_indices<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(rng_state, mutant_indices, population_size);
}

void gpu_set_mutate_indices(cudaStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, bool* __restrict__ mutate_indices, double* __restrict__ crossover_probabilities, size_t population_size, size_t genome_size) {
    gpu_set_mutate_indices<<<dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s>>>(rng_state, mutate_indices, crossover_probabilities, population_size, genome_size);
}
