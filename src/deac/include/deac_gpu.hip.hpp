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
#ifdef DEAC_DEBUG
    #include <stdio.h>
#endif
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
         C[0] = _c[0];
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

__global__
void gpu_normalize_population(double* __restrict__ population, double* __restrict__ normalization, double zeroth_moment, size_t population_size, size_t genome_size) {
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    if (global_idx < population_size*genome_size) {
        population[global_idx] *= zeroth_moment/normalization[global_idx/genome_size];
    }
}

__global__
void gpu_set_fitness(double* __restrict__ fitness, double* __restrict__ isf, double* __restrict__ isf_model, double* __restrict__ isf_error, size_t number_of_timeslices) {
    __shared__ double _f[GPU_BLOCK_SIZE];
    // Set shared local memory _f
    size_t local_idx = hipThreadIdx_x;
    if (local_idx < number_of_timeslices) {
        _f[local_idx] = sycl::pown((isf[local_idx] - isf_model[local_idx])/isf_error[local_idx], 2);
    } else {
        _f[local_idx] = 0.0;
    }

    for (size_t i = 1; i < (number_of_timeslices + GPU_BLOCK_SIZE - 1)/GPU_BLOCK_SIZE; i++) {
        size_t j = GPU_BLOCK_SIZE*i + local_idx;
        if (j < number_of_timeslices) {
            _f[local_idx] += sycl::pown((isf[j] - isf_model[j])/isf_error[j], 2);
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
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    if (global_idx < population_size) {
        fitness[global_idx] += sycl::pown((moment - moments[global_idx])/moment_error, 2);
    }
}

__global__
void gpu_set_fitness_moments_chi_squared(double* __restrict__ fitness, double* __restrict__ moments, double moment, size_t population_size) {
    size_t global_idx = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    if (global_idx < population_size) {
        fitness[global_idx] += sycl::pown(moment - moments[global_idx], 2);
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
void gpu_matrix_multiply_MxN_by_Nx1(double * C, double * A, double * B, size_t N, size_t idx) {
    __shared__ double _c[GPU_BLOCK_SIZE];
    size_t _j = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (_j < N) {
        _c[hipThreadIdx_x] = A[idx*N + _j]*B[_j];
    } else {
        _c[hipThreadIdx_x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _c ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (hipThreadIdx_x < 512) {
            _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (hipThreadIdx_x < 256) {
            _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (hipThreadIdx_x < 128) {
            _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (hipThreadIdx_x < 64) {
                _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 64];
            }
            __syncthreads();
        } 
    }

    if (hipThreadIdx_x < warpSize) {
        warp_reduce(_c, hipThreadIdx_x);
    }

    if (hipThreadIdx_x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_c[hipBlockIdx_x] = _c[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&C[idx], _c[0]);
    }
}

__global__
void gpu_matrix_multiply_LxM_by_MxN(double * C, double * A, double * B, size_t L, size_t M, size_t idx) {
    __shared__ double _c[GPU_BLOCK_SIZE];
    size_t k = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (k < M) {
        size_t _i = idx/L;
        size_t _j = idx - _i*L;
        _c[hipThreadIdx_x] = A[_j*M + k]*B[_i*M + k];
    } else {
        _c[hipThreadIdx_x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _c ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (hipThreadIdx_x < 512) {
            _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (hipThreadIdx_x < 256) {
            _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (hipThreadIdx_x < 128) {
            _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (hipThreadIdx_x < 64) {
                _c[hipThreadIdx_x] += _c[hipThreadIdx_x + 64];
            }
            __syncthreads();
        } 
    }

    if (hipThreadIdx_x < warpSize) {
        warp_reduce(_c, hipThreadIdx_x);
    }

    if (hipThreadIdx_x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_c[hipBlockIdx_x] = _c[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&C[idx], _c[0]);
    }
}

__global__
void gpu_normalize_population(double * population, double * normalization, double zeroth_moment, size_t population_size, size_t genome_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size*genome_size) {
        double _norm = normalization[i/genome_size];
        population[i] *= zeroth_moment/_norm;
    }
}

__global__
void gpu_set_fitness(double * fitness, double * isf, double * isf_model, double * isf_error, size_t number_of_timeslices, size_t idx) {
    __shared__ double _f[GPU_BLOCK_SIZE];
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < number_of_timeslices) {
        _f[hipThreadIdx_x] = pow((isf[i] - isf_model[idx*number_of_timeslices + i])/isf_error[i],2);
    } else {
        _f[hipThreadIdx_x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _f ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (hipThreadIdx_x < 512) {
            _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (hipThreadIdx_x < 256) {
            _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (hipThreadIdx_x < 128) {
            _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (hipThreadIdx_x < 64) {
                _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 64];
            }
            __syncthreads();
        } 
    }

    if (hipThreadIdx_x < warpSize) {
        warp_reduce(_f, hipThreadIdx_x);
    }

    if (hipThreadIdx_x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_f[hipBlockIdx_x] = _f[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&fitness[idx], _f[0]/number_of_timeslices);
    }
}

__global__
void gpu_set_fitness_moments_reduced_chi_squared(double * fitness, double * moments, double moment, double moment_error, size_t population_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        fitness[i] += pow((moment - moments[i])/moment_error,2);
    } 
}

__global__
void gpu_set_fitness_moments_chi_squared(double * fitness, double * moments, double moment, size_t population_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        fitness[i] += pow((moment - moments[i]),2)/moment;
    } 
}

__global__
void gpu_get_minimum_fitness(double * fitness, double * minimum_fitness, size_t population_size) {
    __shared__ double s_minimum[GPU_BLOCK_SIZE];
    if (hipThreadIdx_x < population_size) {
        s_minimum[hipThreadIdx_x] = fitness[hipThreadIdx_x];
    } else {
        s_minimum[hipThreadIdx_x] = fitness[0];
    }

    for (size_t i=0; i<population_size/GPU_BLOCK_SIZE; i++) {
        size_t j = GPU_BLOCK_SIZE*i + hipThreadIdx_x;
        if (j < population_size) {
            s_minimum[hipThreadIdx_x] = fitness[j] < s_minimum[hipThreadIdx_x] ? fitness[j] : s_minimum[hipThreadIdx_x];
        }
    }

    __syncthreads();

    // NEED TO REDUCE _f ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (hipThreadIdx_x < 512) {
            s_minimum[hipThreadIdx_x] = s_minimum[hipThreadIdx_x + 512] < s_minimum[hipThreadIdx_x] ? s_minimum[hipThreadIdx_x + 512] : s_minimum[hipThreadIdx_x];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (hipThreadIdx_x < 256) {
            s_minimum[hipThreadIdx_x] = s_minimum[hipThreadIdx_x + 256] < s_minimum[hipThreadIdx_x] ? s_minimum[hipThreadIdx_x + 256] : s_minimum[hipThreadIdx_x];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (hipThreadIdx_x < 128) {
            s_minimum[hipThreadIdx_x] = s_minimum[hipThreadIdx_x + 128] < s_minimum[hipThreadIdx_x] ? s_minimum[hipThreadIdx_x + 128] : s_minimum[hipThreadIdx_x];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (hipThreadIdx_x < 64) {
                s_minimum[hipThreadIdx_x] = s_minimum[hipThreadIdx_x + 64] < s_minimum[hipThreadIdx_x] ? s_minimum[hipThreadIdx_x + 64] : s_minimum[hipThreadIdx_x];
            }
            __syncthreads();
        } 
    }

    if (hipThreadIdx_x < warpSize) {
        warp_reduce_min(s_minimum, hipThreadIdx_x);
    }

    if (hipThreadIdx_x == 0) {
        *minimum_fitness = s_minimum[0];
    }

}

__global__
void gpu_set_fitness_mean(double * fitness_mean, double * fitness, size_t population_size, size_t idx) {
    __shared__ double _f[GPU_BLOCK_SIZE];
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        _f[hipThreadIdx_x] = fitness[i];
    } else {
        _f[hipThreadIdx_x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _f ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (hipThreadIdx_x < 512) {
            _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (hipThreadIdx_x < 256) {
            _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (hipThreadIdx_x < 128) {
            _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (hipThreadIdx_x < 64) {
                _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 64];
            }
            __syncthreads();
        } 
    }

    if (hipThreadIdx_x < warpSize) {
        warp_reduce(_f, hipThreadIdx_x);
    }

    if (hipThreadIdx_x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_f[hipBlockIdx_x] = _f[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&fitness_mean[idx], _f[0]/population_size);
    }
}

__global__
void gpu_set_fitness_squared_mean(double * fitness_squared_mean, double * fitness, size_t population_size, size_t idx) {
    __shared__ double _f[GPU_BLOCK_SIZE];
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        _f[hipThreadIdx_x] = fitness[i]*fitness[i];
    } else {
        _f[hipThreadIdx_x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _f ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (hipThreadIdx_x < 512) {
            _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (hipThreadIdx_x < 256) {
            _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (hipThreadIdx_x < 128) {
            _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (hipThreadIdx_x < 64) {
                _f[hipThreadIdx_x] += _f[hipThreadIdx_x + 64];
            }
            __syncthreads();
        } 
    }

    if (hipThreadIdx_x < warpSize) {
        warp_reduce(_f, hipThreadIdx_x);
    }

    if (hipThreadIdx_x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_f[hipBlockIdx_x] = _f[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&fitness_squared_mean[idx], _f[0]/population_size);
    }
}

__global__
void gpu_set_population_new(double * population_new, double * population_old, size_t * mutant_indices, double * differential_weights_new, bool * mutate_indices, size_t population_size, size_t genome_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size*genome_size) {
        size_t _i = i/genome_size;
        size_t _j = i - _i*genome_size;
        double F = differential_weights_new[_i];
        size_t mutant_index1 = mutant_indices[3*_i];
        size_t mutant_index2 = mutant_indices[3*_i + 1];
        size_t mutant_index3 = mutant_indices[3*_i + 2];
        bool mutate = mutate_indices[i];
        if (mutate) {
            #ifdef ALLOW_NEGATIVE_SPECTRAL_WEIGHT
                population_new[i] = population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]);
            #else
                population_new[i] = fabs( population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]) );
            #endif
        } else {
            population_new[i] = population_old[i];
        }
    }
}

__global__
void gpu_match_population_zero(double * population_negative_frequency, double * population_positive_frequency, size_t population_size, size_t genome_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        population_negative_frequency[i*genome_size] = population_positive_frequency[i*genome_size];
    }
}

__global__
void gpu_set_rejection_indices(bool * rejection_indices, double * fitness_new, double * fitness_old, size_t population_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        bool accept = fitness_new[i] <= fitness_old[i];
        rejection_indices[i] = accept;
        if (accept) {
            fitness_old[i] = fitness_new[i];
        }
    }
}

__global__
void gpu_swap_control_parameters(double * control_parameter_old, double * control_parameter_new, bool * rejection_indices, size_t population_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        if (rejection_indices[i]) {
            control_parameter_old[i] = control_parameter_new[i];
        }
    }
}

__global__
void gpu_swap_populations(double * population_old, double * population_new, bool * rejection_indices, size_t population_size, size_t genome_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size*genome_size) {
        size_t _i = i/genome_size;
        if (rejection_indices[_i]) {
            population_old[i] = population_new[i];
        }
    }
}

//FIXME
__global__
void gpu_set_crossover_probabilities_new(uint64_t * rng_state, double * crossover_probabilities_new, double * crossover_probabilities_old, double self_adapting_crossover_probability, size_t population_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        if ((gpu_xoshiro256p_next(rng_state + 4*i) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
            crossover_probabilities_new[i] = (gpu_xoshiro256p_next(rng_state + 4*i) >> 11) * 0x1.0p-53;
        } else {
            crossover_probabilities_new[i] = crossover_probabilities_old[i];
        }
    }
}

__global__
void gpu_set_differential_weights_new(uint64_t * rng_state, double * differential_weights_new, double * differential_weights_old, double self_adapting_differential_weight_probability, size_t population_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        if ((gpu_xoshiro256p_next(rng_state + 4*i) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
            differential_weights_new[i] = 2.0*((gpu_xoshiro256p_next(rng_state + 4*i) >> 11) * 0x1.0p-53);
        } else {
            differential_weights_new[i] = differential_weights_old[i];
        }
    }
}

__global__
void gpu_set_mutant_indices(uint64_t * rng_state, size_t * mutant_indices, size_t population_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size) {
        gpu_set_mutant_indices(rng_state + 4*i, mutant_indices + 3*i, i, population_size);
    }
}

__device__
void gpu_set_mutant_indices(uint64_t * rng_state, size_t * mutant_indices, size_t mutant_index0, size_t length) {
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
void gpu_set_mutate_indices(uint64_t * rng_state, bool * mutate_indices, double * crossover_probabilities, size_t population_size, size_t genome_size) {
    size_t i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < population_size*genome_size) {
        size_t _i = i/genome_size;
        mutate_indices[i] = (gpu_xoshiro256p_next(rng_state + 4*i) >> 11) * 0x1.0p-53 < crossover_probabilities[_i];
    }
}

__global__
void gpu_check_minimum_fitness(double * minimum_fitness, double stop_minimum_fitness) {
    assert(*minimum_fitness >= stop_minimum_fitness);
}

// Kernel Launcher
void gpu_dot(sycl::queue q, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
void gpu_get_minimum(sycl::queue q, double* __restrict__ minimum, double* __restrict__ array, size_t N) {
void gpu_normalize_population(sycl::queue q, size_t grid_size, double* __restrict__ population, double* __restrict__ normalization, double zeroth_moment, size_t population_size, size_t genome_size) {
void gpu_set_fitness(sycl::queue q, double* __restrict__ fitness, double* __restrict__ isf, double* __restrict__ isf_model, double* __restrict__ isf_error, size_t number_of_timeslices) {
void gpu_set_fitness_moments_reduced_chi_squared(sycl::queue q, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, double moment_error, size_t population_size) {
void gpu_set_fitness_moments_chi_squared(sycl::queue q, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, size_t population_size) {
void gpu_set_fitness_mean(sycl::queue q, double* __restrict__ fitness_mean, double* __restrict__ fitness, size_t population_size) {
void gpu_set_fitness_squared_mean(sycl::queue q, double* __restrict__ fitness_squared_mean, double* __restrict__ fitness, size_t population_size) {
void gpu_set_population_new(sycl::queue q, size_t grid_size, double* __restrict__ population_new, double* __restrict__ population_old, size_t* __restrict__ mutant_indices, double* __restrict__ differential_weights_new, bool* __restrict__ mutate_indices, size_t population_size, size_t genome_size) {
void gpu_match_population_zero(sycl::queue q, size_t grid_size, double* __restrict__ population_negative_frequency, double* __restrict__ population_positive_frequency, size_t population_size, size_t genome_size) {
void gpu_set_rejection_indices(sycl::queue q, size_t grid_size, bool* __restrict__ rejection_indices, double* __restrict__ fitness_new, double* __restrict__ fitness_old, size_t population_size) {
void gpu_swap_control_parameters(sycl::queue q, size_t grid_size, double* __restrict__ control_parameter_old, double* __restrict__ control_parameter_new, bool* __restrict__ rejection_indices, size_t population_size) {
void gpu_swap_populations(sycl::queue q, size_t grid_size, double* __restrict__ population_old, double* __restrict__ population_new, bool* __restrict__ rejection_indices, size_t population_size, size_t genome_size) {
void gpu_set_crossover_probabilities_new(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ crossover_probabilities_new, double* __restrict__ crossover_probabilities_old, double self_adapting_crossover_probability, size_t population_size) {
void gpu_set_differential_weights_new(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ differential_weights_new, double* __restrict__ differential_weights_old, double self_adapting_differential_weight_probability, size_t population_size) {
void gpu_set_mutant_indices(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, size_t* __restrict__ mutant_indices, size_t population_size) {
void gpu_set_mutate_indices(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, bool* __restrict__ mutate_indices, double* __restrict__ crossover_probabilities, size_t population_size, size_t genome_size) {
#endif
