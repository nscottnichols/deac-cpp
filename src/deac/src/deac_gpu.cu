/**
 * @file deac_gpu.hip.hpp
 * @author Nathan Nichols
 * @date 04.19.2021
 *
 * @brief GPU kernels using CUDA.
 */

#include "deac_gpu.cuh"
#ifdef DEAC_DEBUG
    #include <stdio.h>
#endif
//

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNELS ---------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

#ifdef DEAC_DEBUG
    __global__
    void gpu_check_array(double * _array, int length) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < length) {
            printf("_array[%d]: %e\n", i, _array[i]);
        }
    }
#endif

// GPU Kernel for reduction using warp (uses appropriate warp for NVIDIA vs AMD devices i. e. "portable wave aware code")
__device__ void warp_reduce(volatile double *sdata, unsigned int thread_idx) {
    if (warpSize == 64) { if (GPU_BLOCK_SIZE >= 128) sdata[thread_idx] += sdata[thread_idx + 64]; }
    if (GPU_BLOCK_SIZE >= 64) sdata[thread_idx] += sdata[thread_idx + 32];
    if (GPU_BLOCK_SIZE >= 32) sdata[thread_idx] += sdata[thread_idx + 16];
    if (GPU_BLOCK_SIZE >= 16) sdata[thread_idx] += sdata[thread_idx + 8];
    if (GPU_BLOCK_SIZE >= 8) sdata[thread_idx] += sdata[thread_idx + 4];
    if (GPU_BLOCK_SIZE >= 4) sdata[thread_idx] += sdata[thread_idx + 2];
    if (GPU_BLOCK_SIZE >= 2) sdata[thread_idx] += sdata[thread_idx + 1];
}

__device__ void warp_reduce_min(volatile double *sdata, unsigned int thread_idx) {
    if (warpSize == 64) { if (GPU_BLOCK_SIZE >= 128) sdata[thread_idx] = sdata[thread_idx + 64] < sdata[thread_idx] ? sdata[thread_idx + 64] : sdata[thread_idx]; }
    if (GPU_BLOCK_SIZE >= 64) sdata[thread_idx] = sdata[thread_idx + 32] < sdata[thread_idx] ? sdata[thread_idx + 32] : sdata[thread_idx];
    if (GPU_BLOCK_SIZE >= 32) sdata[thread_idx] = sdata[thread_idx + 16] < sdata[thread_idx] ? sdata[thread_idx + 16] : sdata[thread_idx];
    if (GPU_BLOCK_SIZE >= 16) sdata[thread_idx] = sdata[thread_idx + 8] < sdata[thread_idx] ? sdata[thread_idx + 8] : sdata[thread_idx];
    if (GPU_BLOCK_SIZE >= 8) sdata[thread_idx] = sdata[thread_idx + 4] < sdata[thread_idx] ? sdata[thread_idx + 4] : sdata[thread_idx];
    if (GPU_BLOCK_SIZE >= 4) sdata[thread_idx] = sdata[thread_idx + 2] < sdata[thread_idx] ? sdata[thread_idx + 2] : sdata[thread_idx];
    if (GPU_BLOCK_SIZE >= 2) sdata[thread_idx] = sdata[thread_idx + 1] < sdata[thread_idx] ? sdata[thread_idx + 1] : sdata[thread_idx];
}

__global__
void gpu_matrix_multiply_MxN_by_Nx1(double * C, double * A, double * B, int N, int idx) {
    __shared__ double _c[GPU_BLOCK_SIZE];
    int _j = blockDim.x * blockIdx.x + threadIdx.x;
    if (_j < N) {
        _c[threadIdx.x] = A[idx*N + _j]*B[_j];
    } else {
        _c[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _c ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (threadIdx.x < 512) {
            _c[threadIdx.x] += _c[threadIdx.x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) {
            _c[threadIdx.x] += _c[threadIdx.x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) {
            _c[threadIdx.x] += _c[threadIdx.x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (threadIdx.x < 64) {
                _c[threadIdx.x] += _c[threadIdx.x + 64];
            }
            __syncthreads();
        } 
    }

    if (threadIdx.x < warpSize) {
        warp_reduce(_c, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_c[blockIdx.x] = _c[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&C[idx], _c[0]);
    }
}

__global__
void gpu_matrix_multiply_LxM_by_MxN(double * C, double * A, double * B, int L, int M, int idx) {
    __shared__ double _c[GPU_BLOCK_SIZE];
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < M) {
        int _i = idx/L;
        int _j = idx - _i*L;
        _c[threadIdx.x] = A[_j*M + k]*B[_i*M + k];
    } else {
        _c[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _c ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (threadIdx.x < 512) {
            _c[threadIdx.x] += _c[threadIdx.x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) {
            _c[threadIdx.x] += _c[threadIdx.x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) {
            _c[threadIdx.x] += _c[threadIdx.x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (threadIdx.x < 64) {
                _c[threadIdx.x] += _c[threadIdx.x + 64];
            }
            __syncthreads();
        } 
    }

    if (threadIdx.x < warpSize) {
        warp_reduce(_c, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_c[blockIdx.x] = _c[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&C[idx], _c[0]);
    }
}

__global__
void gpu_normalize_population(double * population, double * normalization, double zeroth_moment, int population_size, int genome_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < population_size*genome_size) {
        double _norm = normalization[i/genome_size];
        population[i] *= zeroth_moment/_norm;
    }
}

__global__
void gpu_set_fitness(double * fitness, double * isf, double * isf_model, double * isf_error, int number_of_timeslices, int idx) {
    __shared__ double _f[GPU_BLOCK_SIZE];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < number_of_timeslices) {
        _f[threadIdx.x] = pow((isf[i] - isf_model[idx*number_of_timeslices + i])/isf_error[i],2);
    } else {
        _f[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _f ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (threadIdx.x < 512) {
            _f[threadIdx.x] += _f[threadIdx.x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) {
            _f[threadIdx.x] += _f[threadIdx.x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) {
            _f[threadIdx.x] += _f[threadIdx.x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (threadIdx.x < 64) {
                _f[threadIdx.x] += _f[threadIdx.x + 64];
            }
            __syncthreads();
        } 
    }

    if (threadIdx.x < warpSize) {
        warp_reduce(_f, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_f[blockIdx.x] = _f[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&fitness[idx], _f[0]/number_of_timeslices);
    }
}

__global__
void gpu_set_fitness_moments_reduced_chi_squared(double * fitness, double * moments, double moment, double moment_error, int population_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < population_size) {
        fitness[i] += pow((moment - moments[i])/moment_error,2);
    } 
}

__global__
void gpu_set_fitness_moments_chi_squared(double * fitness, double * moments, double moment, int population_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < population_size) {
        fitness[i] += pow((moment - moments[i]),2)/moment;
    } 
}

__global__
void gpu_get_minimum_fitness(double * fitness, double * minimum_fitness, int population_size) {
    __shared__ double s_minimum[GPU_BLOCK_SIZE];
    if (threadIdx.x < population_size) {
        s_minimum[threadIdx.x] = fitness[threadIdx.x];
    } else {
        s_minimum[threadIdx.x] = fitness[0];
    }

    for (int i=0; i<population_size/GPU_BLOCK_SIZE; i++) {
        int j = GPU_BLOCK_SIZE*i + threadIdx.x;
        if (j < population_size) {
            s_minimum[threadIdx.x] = fitness[j] < s_minimum[threadIdx.x] ? fitness[j] : s_minimum[threadIdx.x];
        }
    }

    __syncthreads();

    // NEED TO REDUCE _f ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (threadIdx.x < 512) {
            s_minimum[threadIdx.x] = s_minimum[threadIdx.x + 512] < s_minimum[threadIdx.x] ? s_minimum[threadIdx.x + 512] : s_minimum[threadIdx.x];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) {
            s_minimum[threadIdx.x] = s_minimum[threadIdx.x + 256] < s_minimum[threadIdx.x] ? s_minimum[threadIdx.x + 256] : s_minimum[threadIdx.x];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) {
            s_minimum[threadIdx.x] = s_minimum[threadIdx.x + 128] < s_minimum[threadIdx.x] ? s_minimum[threadIdx.x + 128] : s_minimum[threadIdx.x];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (threadIdx.x < 64) {
                s_minimum[threadIdx.x] = s_minimum[threadIdx.x + 64] < s_minimum[threadIdx.x] ? s_minimum[threadIdx.x + 64] : s_minimum[threadIdx.x];
            }
            __syncthreads();
        } 
    }

    if (threadIdx.x < warpSize) {
        warp_reduce_min(s_minimum, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        *minimum_fitness = s_minimum[0];
    }

}

__global__
void gpu_set_fitness_mean(double * fitness_mean, double * fitness, int population_size, int idx) {
    __shared__ double _f[GPU_BLOCK_SIZE];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < population_size) {
        _f[threadIdx.x] = fitness[i];
    } else {
        _f[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _f ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (threadIdx.x < 512) {
            _f[threadIdx.x] += _f[threadIdx.x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) {
            _f[threadIdx.x] += _f[threadIdx.x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) {
            _f[threadIdx.x] += _f[threadIdx.x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (threadIdx.x < 64) {
                _f[threadIdx.x] += _f[threadIdx.x + 64];
            }
            __syncthreads();
        } 
    }

    if (threadIdx.x < warpSize) {
        warp_reduce(_f, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_f[blockIdx.x] = _f[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&fitness_mean[idx], _f[0]/population_size);
    }
}

__global__
void gpu_set_fitness_standard_deviation(double * fitness_standard_deviation, double * fitness_mean, double * fitness, int population_size, int idx) {
    __shared__ double _f[GPU_BLOCK_SIZE];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < population_size) {
        _f[threadIdx.x] = pow(fitness[i] - fitness_mean[idx],2);
    } else {
        _f[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // NEED TO REDUCE _f ON SHARED MEMORY AND ADD TO GLOBAL isf
    if (GPU_BLOCK_SIZE >= 1024) {
        if (threadIdx.x < 512) {
            _f[threadIdx.x] += _f[threadIdx.x + 512];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 512) {
        if (threadIdx.x < 256) {
            _f[threadIdx.x] += _f[threadIdx.x + 256];
        }
        __syncthreads();
    } 

    if (GPU_BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) {
            _f[threadIdx.x] += _f[threadIdx.x + 128];
        }
        __syncthreads();
    } 

    if (warpSize == 32) {
        if (GPU_BLOCK_SIZE >= 128) {
            if (threadIdx.x < 64) {
                _f[threadIdx.x] += _f[threadIdx.x + 64];
            }
            __syncthreads();
        } 
    }

    if (threadIdx.x < warpSize) {
        warp_reduce(_f, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        //NOTE: May see some performance gain here if temporarily store results
        // to some global device variable and then launch a separate kernel to
        // again reduce those results i.e.
        // tmp_f[blockIdx.x] = _f[0];
        // ^-- reduce on this, but this code may get too bloated
        atomicAdd(&fitness_mean[idx], _f[0]/population_size);
    }
}

__global__
void gpu_set_fitness_standard_deviation_sqrt(double * fitness_standard_deviation, int max_generations) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < max_generations) {
        fitness_standard_deviation[i] = sqrt(fitness_standard_deviation[i]);
    }
}

__global__
void gpu_set_population_new(double * population_new, double * population_old, int * mutant_indices, double * differential_weights_new, bool * mutate_indices, int population_size, int genome_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < population_size*genome_size) {
        int _i = i/genome_size;
        int _j = i - _i*genome_size;
        double F = differential_weights_new[_i];
        int mutant_index1 = mutant_indices[3*_i];
        int mutant_index2 = mutant_indices[3*_i + 1];
        int mutant_index3 = mutant_indices[3*_i + 2];
        bool mutate = mutate_indices[i];
        if (mutate) {
            population_new[i] = fabs( 
                population_old[mutant_index1*genome_size + _j] + F*(
                        population_old[mutant_index2*genome_size + _j] -
                        population_old[mutant_index3*genome_size + _j]));
        } else {
            population_new[i] = population_old[i];
        }
    }
}

__global__
void gpu_set_rejection_indices(bool * rejection_indices, double * fitness_new, double * fitness_old, int population_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < population_size) {
        bool accept = fitness_new[i] <= fitness_old[i];
        rejection_indices[i] = accept;
        if (accept) {
            fitness_old[i] = fitness_new[i];
        }
    }
}

__global__
void gpu_swap_control_parameters(double * control_parameter_old, double * control_parameter_new, bool * rejection_indices, int population_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < population_size) {
        if (rejection_indices[i]) {
            control_parameter_old[i] = control_parameter_new[i];
        }
    }
}

__global__
void gpu_swap_populations(double * population_old, double * population_new, bool * rejection_indices, int population_size, int genome_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < population_size*genome_size) {
        int _i = i/genome_size;
        if (rejection_indices[_i]) {
            population_old[i] = population_new[i];
        }
    }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNEL WRAPPER --------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
namespace cuda_wrapper {
    #ifdef DEAC_DEBUG
        void gpu_check_array_wrapper(dim3 grid_size, dim3 group_size, double * _array, int length) {
            gpu_check_array <<<grid_size, group_size, 0, 0>>> ( 
                    _array, length
                    );
        }
    
        void gpu_check_array_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * _array, int length) {
            gpu_check_array <<<grid_size, group_size, 0, stream>>> ( 
                    _array, length
                    );
        }
    #endif

    void gpu_matrix_multiply_MxN_by_Nx1_wrapper(dim3 grid_size, dim3 group_size, double * C, double * A, double * B, int N, int idx) {
        gpu_matrix_multiply_MxN_by_Nx1 <<<grid_size, group_size, 0, 0>>> ( 
                C, A, B, N, idx
                );
    }
    void gpu_matrix_multiply_MxN_by_Nx1_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * C, double * A, double * B, int N, int idx) {
        gpu_matrix_multiply_MxN_by_Nx1 <<<grid_size, group_size, 0, stream>>> ( 
                C, A, B, N, idx
                );
    }
    
    void gpu_matrix_multiply_LxM_by_MxN_wrapper(dim3 grid_size, dim3 group_size, double * C, double * A, double * B, int L, int M, int idx) {
        gpu_matrix_multiply_LxM_by_MxN <<<grid_size, group_size, 0, 0>>> ( 
                C, A, B, L, M, idx
                );
    }
    void gpu_matrix_multiply_LxM_by_MxN_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * C, double * A, double * B, int L, int M, int idx) {
        gpu_matrix_multiply_LxM_by_MxN <<<grid_size, group_size, 0, stream>>> ( 
                C, A, B, L, M, idx
                );
    }
    
    void gpu_normalize_population_wrapper(dim3 grid_size, dim3 group_size, double * population, double * normalization, double zeroth_moment, int population_size, int genome_size) {
        gpu_normalize_population <<<grid_size, group_size, 0, 0>>> ( 
                population, normalization, zeroth_moment, population_size, genome_size
                );
    }
    void gpu_normalize_population_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * population, double * normalization, double zeroth_moment, int population_size, int genome_size) {
        gpu_normalize_population <<<grid_size, group_size, 0, stream>>> ( 
                population, normalization, zeroth_moment, population_size, genome_size
                );
    }
    
    void gpu_set_fitness_wrapper(dim3 grid_size, dim3 group_size, double * fitness, double * isf, double * isf_model, double * isf_error, int number_of_timeslices, int idx) {
        gpu_set_fitness <<<grid_size, group_size, 0, 0>>> ( 
                fitness, isf, isf_model, isf_error, number_of_timeslices, idx
                );
    }
    void gpu_set_fitness_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * fitness, double * isf, double * isf_model, double * isf_error, int number_of_timeslices, int idx) {
        gpu_set_fitness <<<grid_size, group_size, 0, stream>>> ( 
                fitness, isf, isf_model, isf_error, number_of_timeslices, idx
                );
    }
    
    void gpu_set_fitness_moments_reduced_chi_squared_wrapper(dim3 grid_size, dim3 group_size, double * fitness, double * moments, double moment, double moment_error, int population_size) {
        gpu_set_fitness_moments_reduced_chi_squared <<<grid_size, group_size, 0, 0>>> ( 
                fitness, moments, moment, moment_error, population_size
                );
    }
    void gpu_set_fitness_moments_reduced_chi_squared_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * fitness, double * moments, double moment, double moment_error, int population_size) {
        gpu_set_fitness_moments_reduced_chi_squared <<<grid_size, group_size, 0, stream>>> ( 
                fitness, moments, moment, moment_error, population_size
                );
    }
    
    void gpu_set_fitness_moments_chi_squared_wrapper(dim3 grid_size, dim3 group_size, double * fitness, double * moments, double moment, int population_size) {
        gpu_set_fitness_moments_chi_squared <<<grid_size, group_size, 0, 0>>> ( 
                fitness, moments, moment, population_size
                );
    }
    void gpu_set_fitness_moments_chi_squared_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * fitness, double * moments, double moment, int population_size) {
        gpu_set_fitness_moments_chi_squared <<<grid_size, group_size, 0, stream>>> ( 
                fitness, moments, moment, population_size
                );
    }
    
    void gpu_get_minimum_fitness_wrapper(dim3 grid_size, dim3 group_size, double * fitness, double * minimum_fitness, int population_size) {
        gpu_get_minimum_fitness <<<grid_size, group_size, 0, 0>>> ( 
                fitness, minimum_fitness, population_size
                );
    }
    void gpu_get_minimum_fitness_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * fitness, double * minimum_fitness, int population_size) {
        gpu_get_minimum_fitness <<<grid_size, group_size, 0, stream>>> ( 
                fitness, minimum_fitness, population_size
                );
    }
    
    void gpu_set_fitness_mean_wrapper(dim3 grid_size, dim3 group_size, double * fitness_mean, double * fitness, int population_size, int idx) {
        gpu_set_fitness_mean <<<grid_size, group_size, 0, 0>>> ( 
                fitness_mean, fitness, population_size, idx
                );
    }
    void gpu_set_fitness_mean_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * fitness_mean, double * fitness, int population_size, int idx) {
        gpu_set_fitness_mean <<<grid_size, group_size, 0, stream>>> ( 
                fitness_mean, fitness, population_size, idx
                );
    }
    
    void gpu_set_fitness_standard_deviation_wrapper(dim3 grid_size, dim3 group_size, double * fitness_standard_deviation, double * fitness_mean, double * fitness, int population_size, int idx) {
        gpu_set_fitness_standard_deviation <<<grid_size, group_size, 0, 0>>> ( 
                fitness_standard_deviation, fitness_mean, fitness, population_size, idx
                );
    }
    void gpu_set_fitness_standard_deviation_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * fitness_standard_deviation, double * fitness_mean, double * fitness, int population_size, int idx) {
        gpu_set_fitness_standard_deviation <<<grid_size, group_size, 0, stream>>> ( 
                fitness_standard_deviation, fitness_mean, fitness, population_size, idx
                );
    }
    
    void gpu_set_fitness_standard_deviation_sqrt_wrapper(dim3 grid_size, dim3 group_size, double * fitness_standard_deviation, int max_generations) {
        gpu_set_fitness_standard_deviation_sqrt <<<grid_size, group_size, 0, 0>>> ( 
                fitness_standard_deviation, max_generations
                );
    }
    void gpu_set_fitness_standard_deviation_sqrt_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * fitness_standard_deviation, int max_generations) {
        gpu_set_fitness_standard_deviation_sqrt <<<grid_size, group_size, 0, stream>>> ( 
                fitness_standard_deviation, max_generations
                );
    }
    
    void gpu_set_population_new_wrapper(dim3 grid_size, dim3 group_size, double * population_new, double * population_old, int * mutant_indices, double * differential_weights_new, bool * mutate_indices, int population_size, int genome_size) {
        gpu_set_population_new <<<grid_size, group_size, 0, 0>>> ( 
                population_new, population_old, mutant_indices, differential_weights_new, mutate_indices, population_size, genome_size
                );
    }
    void gpu_set_population_new_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * population_new, double * population_old, int * mutant_indices, double * differential_weights_new, bool * mutate_indices, int population_size, int genome_size) {
        gpu_set_population_new <<<grid_size, group_size, 0, stream>>> ( 
                population_new, population_old, mutant_indices, differential_weights_new, mutate_indices, population_size, genome_size
                );
    }
    
    void gpu_set_rejection_indices_wrapper(dim3 grid_size, dim3 group_size, bool * rejection_indices, double * fitness_new, double * fitness_old, int population_size) {
        gpu_set_rejection_indices <<<grid_size, group_size, 0, 0>>> ( 
                rejection_indices, fitness_new, fitness_old, population_size
                );
    }
    void gpu_set_rejection_indices_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, bool * rejection_indices, double * fitness_new, double * fitness_old, int population_size) {
        gpu_set_rejection_indices <<<grid_size, group_size, 0, stream>>> ( 
                rejection_indices, fitness_new, fitness_old, population_size
                );
    }
    
    void gpu_swap_control_parameters_wrapper(dim3 grid_size, dim3 group_size, double * control_parameter_old, double * control_parameter_new, bool * rejection_indices, int population_size) {
        gpu_swap_control_parameters <<<grid_size, group_size, 0, 0>>> ( 
                control_parameter_old, control_parameter_new, rejection_indices, population_size
                );
    }
    void gpu_swap_control_parameters_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * control_parameter_old, double * control_parameter_new, bool * rejection_indices, int population_size) {
        gpu_swap_control_parameters <<<grid_size, group_size, 0, stream>>> ( 
                control_parameter_old, control_parameter_new, rejection_indices, population_size
                );
    }
    
    void gpu_swap_populations_wrapper(dim3 grid_size, dim3 group_size, double * population_old, double * population_new, bool * rejection_indices, int population_size, int genome_size) {
        gpu_swap_populations <<<grid_size, group_size, 0, 0>>> ( 
                population_old, population_new, rejection_indices, population_size, genome_size
                );
    }
    void gpu_swap_populations_wrapper(dim3 grid_size, dim3 group_size, cudaStream_t stream, double * population_old, double * population_new, bool * rejection_indices, int population_size, int genome_size) {
        gpu_swap_populations <<<grid_size, group_size, 0, stream>>> ( 
                population_old, population_new, rejection_indices, population_size, genome_size
                );
    }
}
