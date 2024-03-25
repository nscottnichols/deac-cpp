/**
 * @file deac_gpu.hip.hpp
 * @author Nathan Nichols
 * @date 04.19.2021
 *
 * @brief GPU kernels using HIP.
 */

#ifndef DEAC_GPU_HIP_H 
#define DEAC_GPU_HIP_H

#include "common_gpu.hpp"
#include <stdint.h>


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
    size_t local_idx = hipThreadIdx_x;
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
    size_t local_idx = hipThreadIdx_x;
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
    size_t local_idx = hipThreadIdx_x;
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
         C[0] += _c[0]; //FIXME should do C[0] = _c[0] + scale_factor*C[0] here probably
    }
}

__global__ void gpu_matmul_simple(int m, int n, int k, double alpha, double* __restrict__ A, int lda, double* __restrict__ B, int ldb, double beta, double* __restrict__ C, int ldc) {
    int row = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (row < m && col < n) {
        double sum = 0.0;
        for (int e = 0; e < k; e++) {
            sum += A[row + e * lda] * B[e + col * ldb];
        }
        C[row + col * ldc] = alpha * sum + beta * C[row + col * ldc];
    }
}

__global__ void gpu_matmul(int m, int n, int k, double alpha, double* __restrict__ A, int lda, double* __restrict__ B, int ldb, double beta, double* __restrict__ C, int ldc) {
    int bx = hipBlockIdx_x, by = hipBlockIdx_y;
    int tx = hipThreadIdx_x, ty = hipThreadIdx_y;

    // Identify the row and column of the C element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    double Cvalue = 0.0;

    // Loop over the A and B tiles required to compute the C element
    for (int t = 0; t < (k-1)/TILE_WIDTH + 1; ++t) {
        __shared__ double As[TILE_WIDTH][TILE_WIDTH];
        __shared__ double Bs[TILE_WIDTH][TILE_WIDTH];

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (row < m && t*TILE_WIDTH+tx < k)
            As[ty][tx] = A[row + lda * (t*TILE_WIDTH+tx)];
        else
            As[ty][tx] = 0.0;

        if (t*TILE_WIDTH+ty < k && col < n)
            Bs[ty][tx] = B[(t*TILE_WIDTH+ty) + ldb * col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads(); // Make sure the matrices are loaded before starting the computation

        // Multiply the two matrices together; each thread computes one element of the block sub-matrix
        for (int e = 0; e < TILE_WIDTH; ++e) {
            Cvalue += As[ty][e] * Bs[e][tx];
        }

        __syncthreads(); // Make sure that all threads are done computing before loading the next set of tiles
    }

    if (row < m && col < n)
        C[row + ldc * col] = alpha * Cvalue + beta * C[row + ldc * col];
}

__global__ void gpu_deac_gemv_simple(int m, int n, double alpha, double* __restrict__ A, int lda, double* __restrict__ x, int incx, double beta, double* __restrict__ y, int incy) {
    int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row < m) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[row + j*lda] * x[j*incx];
        }
        y[row*incy] = alpha * sum + beta * y[row*incy];
    }
}

__global__ void gpu_deac_gemv_atomic(int m, int n, double alpha, double* __restrict__ A, int lda, double* __restrict__ x, int incx, double beta, double* __restrict__ y, int incy) {
    __shared__ double shared_x[TILE_WIDTH];
    int col = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (col < n) {
        shared_x[hipThreadIdx_x] = x[col * incx];
    }
    __syncthreads();

    if (col < n) {
        for (int i = 0; i < m; i++) {
            double Aval = A[i + col * lda];
            atomicAdd(&y[i * incy], alpha * Aval * shared_x[hipThreadIdx_x]);
        }
    }
}

__global__ void gpu_deac_gemv(int m, int n, double alpha, double* __restrict__ A, int lda, double* __restrict__ x, int incx, double beta, double* __restrict__ y, int incy) {
    __shared__ double As[TILE_WIDTH][TILE_WIDTH];
    int tx = hipThreadIdx_x;
    int by = hipBlockIdx_y, ty = hipThreadIdx_y;
    int row = by * hipBlockDim_y + ty;

    double sum = 0.0;
    if (row < m) {
        for (int i = 0; i < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++i) {
            if (i*TILE_WIDTH + tx < n && row < m) {
                As[ty][tx] = A[row + (i*TILE_WIDTH + tx) * lda];
            } else {
                As[ty][tx] = 0.0;
            }
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                if (i*TILE_WIDTH + k < n) {
                    sum += As[ty][k] * x[(i*TILE_WIDTH + k)*incx];
                }
            }
            __syncthreads();
        }
        if (beta == 0.0) {
            y[row * incy] = alpha * sum;
        } else {
            y[row * incy] = alpha * sum + beta * y[row * incy];
        }
    }
}


__global__
void gpu_get_minimum(double* __restrict__ minimum, double* __restrict__ array, size_t N) {
    // finds minimum of array with length N
    // Shared Local Memory _c
    __shared__ double _c[GPU_BLOCK_SIZE];
    // Set shared local memory _c
    size_t local_idx = hipThreadIdx_x;
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

__global__ void gpu_deac_dgmmDiv1D(double* __restrict__ matrix, double* __restrict__ vector, size_t rows, size_t cols) {
    size_t idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; // 1D index for the entire matrix
    if (idx < rows * cols) { // Ensure we do not go out of bounds
        size_t row = idx % rows; // Calculate the row index
        //size_t col = idx / rows; // Calculate the column index, assuming column-major order
        double reciprocal = 1.0/vector[row];
        // Perform the division operation
        matrix[idx] *= reciprocal;
    }
}

__global__ void gpu_deac_reduced_chi_squared(const double* __restrict__ calculated_data, const double* __restrict__ observed_data, const double* __restrict__ standard_deviations, double* __restrict__ reduced_chi_squared, size_t m, size_t n, size_t ddof, double beta) {
    __shared__ double sdata[GPU_BLOCK_SIZE]; // Use static shared memory for reduction
    size_t row = hipBlockIdx_x;
    size_t tid = hipThreadIdx_x;
    double sum = 0.0;

    // Loop over all elements assigned to this thread
    for (size_t idx = tid; idx < n; idx += hipBlockDim_x) {
        double O = observed_data[idx];
        double E = calculated_data[row + idx * m]; // Access pattern for column-major storage
        double sigma = standard_deviations[idx];
        double term = (O - E) / sigma;
        sum += term * term;
    }

    // Load the thread's sum into shared memory and synchronize
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (size_t s = hipBlockDim_x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Have the first thread in the block write the result for this row to global memory
    if (tid == 0) {
        reduced_chi_squared[row] = sdata[0]/(n - ddof) + beta*reduced_chi_squared[row];
    }
}

__global__
void gpu_set_fitness_mean(double* __restrict__ fitness_mean, double* __restrict__ fitness, size_t population_size) {
    __shared__ double _fm[GPU_BLOCK_SIZE];
    // Set shared local memory _fm
    size_t local_idx = hipThreadIdx_x;
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
    size_t local_idx = hipThreadIdx_x;
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
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    if (global_idx < population_size*genome_size) {
        //size_t _i = global_idx/genome_size;
        //size_t _j = global_idx - _i*genome_size;
        size_t _i = global_idx % population_size; // row
        size_t _j = global_idx / population_size; // col
        double F = differential_weights_new[_i];
        size_t mutant_index1 = mutant_indices[3*_i];
        size_t mutant_index2 = mutant_indices[3*_i + 1];
        size_t mutant_index3 = mutant_indices[3*_i + 2];
        bool mutate = mutate_indices[global_idx];
        if (mutate) {
            #ifdef ALLOW_NEGATIVE_SPECTRAL_WEIGHT
                //population_new[global_idx] = population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]);
                population_new[global_idx] = population_old[population_size*_j + mutant_index1] + F*(population_old[population_size*_j + mutant_index2] - population_old[population_size*_j + mutant_index3]);
            #else
                //population_new[global_idx] = fabs(population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]));
                population_new[global_idx] = fabs(population_old[population_size*_j + mutant_index1] + F*(population_old[population_size*_j + mutant_index2] - population_old[population_size*_j + mutant_index3]));
            #endif
        } else {
            population_new[global_idx] = population_old[global_idx];
        }
    }
}


__global__
void gpu_match_population_zero(double* __restrict__ population_negative_frequency, double* __restrict__ population_positive_frequency, size_t population_size, size_t genome_size) {
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    if (global_idx < population_size) {
        //population_negative_frequency[global_idx*genome_size] = population_positive_frequency[global_idx*genome_size];
        population_negative_frequency[global_idx] = population_positive_frequency[global_idx]; // Column-major
    }
}

__global__
void gpu_set_rejection_indices(bool* __restrict__ rejection_indices, double* __restrict__ fitness_new, double* __restrict__ fitness_old, size_t population_size) {
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
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
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    if (global_idx < population_size) {
        if (rejection_indices[global_idx]) {
            control_parameter_old[global_idx] = control_parameter_new[global_idx];
        }
    }
}

__global__
void gpu_swap_populations(double* __restrict__ population_old, double* __restrict__ population_new, bool* __restrict__ rejection_indices, size_t population_size, size_t genome_size) {
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    if (global_idx < population_size*genome_size) {
        size_t _i = global_idx/genome_size;
        if (rejection_indices[_i]) {
            population_old[global_idx] = population_new[global_idx];
        }
    }
}

__global__
void gpu_set_crossover_probabilities_new(uint64_t* __restrict__ rng_state, double* __restrict__ crossover_probabilities_new, double* __restrict__ crossover_probabilities_old, double self_adapting_crossover_probability, size_t population_size) {
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
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
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
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
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    if (global_idx < population_size) {
        gpu_set_mutant_indices(rng_state + 4*global_idx, mutant_indices + 3*global_idx, global_idx, population_size);
    }
}

__global__
void gpu_set_mutate_indices(uint64_t* __restrict__ rng_state, bool* __restrict__ mutate_indices, double* __restrict__ crossover_probabilities, size_t population_size, size_t genome_size) {
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    if (global_idx < population_size*genome_size) {
        //size_t _i = global_idx/genome_size;
        size_t _i = global_idx % population_size;
        mutate_indices[global_idx] = (gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < crossover_probabilities[_i];
    }
}

// Kernel Launcher
void gpu_dot(hipStream_t s, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
    hipLaunchKernelGGL(gpu_dot,
            dim3(1), dim3(GPU_BLOCK_SIZE), 0, s,
            C, B, A, N);
}

void gpu_matmul(hipStream_t s, int m, int n, int k, double alpha, double* __restrict__ A, double* __restrict__ B, double beta, double* __restrict__ C) {
    hipLaunchKernelGGL(gpu_matmul, dim3((n + TILE_WIDTH - 1) / TILE_WIDTH, (m + TILE_WIDTH - 1) / TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH), 0, s, m, n, k, alpha, A, m, B, k, beta, C, m);
}

void gpu_deac_gemv(hipStream_t s, int m, int n, double alpha, double* __restrict__ A, double* __restrict__ x, double beta, double* __restrict__ y) {
    //hipLaunchKernelGGL(gpu_deac_gemv_simple, dim3((m + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE), dim3(GPU_BLOCK_SIZE), 0, s, m, n, alpha, A, m, x, 1, beta, y, 1);
    //hipLaunchKernelGGL(gpu_deac_gemv_atomic, dim3((n + TILE_WIDTH - 1) / TILE_WIDTH), dim3(TILE_WIDTH), 0, s, m, n, alpha, A, m, x, 1, beta, y, 1);
    hipLaunchKernelGGL(gpu_deac_gemv, dim3((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH), 0, s, m, n, alpha, A, m, x, 1, beta, y, 1);
}


void gpu_get_minimum(hipStream_t s, double* __restrict__ minimum, double* __restrict__ array, size_t N) {
    hipLaunchKernelGGL(gpu_get_minimum,
            dim3(1), dim3(GPU_BLOCK_SIZE), 0, s,
            minimum, array, N);
}

void gpu_deac_dgmmDiv1D(hipStream_t s, double* __restrict__ matrix, double* __restrict__ vector, size_t rows, size_t cols) {
    hipLaunchKernelGGL(gpu_deac_dgmmDiv1D, dim3((rows*cols + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE), dim3(GPU_BLOCK_SIZE), 0, s, matrix, vector, rows, cols);
}

void gpu_deac_reduced_chi_squared(hipStream_t s, const double* __restrict__ calculated_data, const double* __restrict__ observed_data, const double* __restrict__ standard_deviations, double* __restrict__ reduced_chi_squared, size_t m, size_t n, size_t ddof, double beta) {
    hipLaunchKernelGGL(gpu_deac_reduced_chi_squared, dim3(m), dim3(GPU_BLOCK_SIZE), 0, s, calculated_data, observed_data, standard_deviations, reduced_chi_squared, m, n, ddof, beta);
}

void gpu_set_fitness_mean(hipStream_t s, double* __restrict__ fitness_mean, double* __restrict__ fitness, size_t population_size) {
    hipLaunchKernelGGL(gpu_set_fitness_mean,
            dim3(1), dim3(GPU_BLOCK_SIZE), 0, s,
            fitness_mean, fitness, population_size);
}

void gpu_set_fitness_squared_mean(hipStream_t s, double* __restrict__ fitness_squared_mean, double* __restrict__ fitness, size_t population_size) {
    hipLaunchKernelGGL(gpu_set_fitness_squared_mean,
            dim3(1), dim3(GPU_BLOCK_SIZE), 0, s,
            fitness_squared_mean, fitness, population_size);
}

void gpu_set_population_new(hipStream_t s, size_t grid_size, double* __restrict__ population_new, double* __restrict__ population_old, size_t* __restrict__ mutant_indices, double* __restrict__ differential_weights_new, bool* __restrict__ mutate_indices, size_t population_size, size_t genome_size) {
    hipLaunchKernelGGL(gpu_set_population_new,
            dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s,
            population_new, population_old, mutant_indices, differential_weights_new, mutate_indices, population_size, genome_size);
}

void gpu_match_population_zero(hipStream_t s, size_t grid_size, double* __restrict__ population_negative_frequency, double* __restrict__ population_positive_frequency, size_t population_size, size_t genome_size) {
    hipLaunchKernelGGL(gpu_match_population_zero,
            dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s,
            population_negative_frequency, population_positive_frequency, population_size, genome_size);
}

void gpu_set_rejection_indices(hipStream_t s, size_t grid_size, bool* __restrict__ rejection_indices, double* __restrict__ fitness_new, double* __restrict__ fitness_old, size_t population_size) {
    hipLaunchKernelGGL(gpu_set_rejection_indices,
            dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s,
            rejection_indices, fitness_new, fitness_old, population_size);
}

void gpu_swap_control_parameters(hipStream_t s, size_t grid_size, double* __restrict__ control_parameter_old, double* __restrict__ control_parameter_new, bool* __restrict__ rejection_indices, size_t population_size) {
    hipLaunchKernelGGL(gpu_swap_control_parameters,
            dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s,
            control_parameter_old, control_parameter_new, rejection_indices, population_size);
}

void gpu_swap_populations(hipStream_t s, size_t grid_size, double* __restrict__ population_old, double* __restrict__ population_new, bool* __restrict__ rejection_indices, size_t population_size, size_t genome_size) {
    hipLaunchKernelGGL(gpu_swap_populations,
            dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s,
            population_old, population_new, rejection_indices, population_size, genome_size);
}

void gpu_set_crossover_probabilities_new(hipStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ crossover_probabilities_new, double* __restrict__ crossover_probabilities_old, double self_adapting_crossover_probability, size_t population_size) {
    hipLaunchKernelGGL(gpu_set_crossover_probabilities_new,
            dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s,
            rng_state, crossover_probabilities_new, crossover_probabilities_old, self_adapting_crossover_probability, population_size);
}

void gpu_set_differential_weights_new(hipStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ differential_weights_new, double* __restrict__ differential_weights_old, double self_adapting_differential_weight_probability, size_t population_size) {
    hipLaunchKernelGGL(gpu_set_differential_weights_new,
            dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s,
            rng_state, differential_weights_new, differential_weights_old, self_adapting_differential_weight_probability, population_size);
}

void gpu_set_mutant_indices(hipStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, size_t* __restrict__ mutant_indices, size_t population_size) {
    hipLaunchKernelGGL(gpu_set_mutant_indices,
            dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s,
            rng_state, mutant_indices, population_size);
}

void gpu_set_mutate_indices(hipStream_t s, size_t grid_size, uint64_t* __restrict__ rng_state, bool* __restrict__ mutate_indices, double* __restrict__ crossover_probabilities, size_t population_size, size_t genome_size) {
    hipLaunchKernelGGL(gpu_set_mutate_indices,
            dim3(grid_size), dim3(GPU_BLOCK_SIZE), 0, s,
            rng_state, mutate_indices, crossover_probabilities, population_size, genome_size);
}

#ifdef USE_BLAS
    void gpu_blas_gemv(hipblasHandle_t handle, int m, int n, double alpha, double* A, double* B, double beta, double* C) {
        GPU_BLAS_ASSERT(hipblasDgemv(handle, HIPBLAS_OP_N, m, n, &alpha, A, m, B, 1, &beta, C, 1));
    }

    void gpu_blas_gemm(hipblasHandle_t handle, int m, int n, int k, double alpha, double* A, double* B, double beta, double* C) {
        GPU_BLAS_ASSERT(hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m));
    }
#endif
#endif
