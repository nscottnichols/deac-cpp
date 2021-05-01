#ifndef COMMON_GPU_H 
#define COMMON_GPU_H
#include <cassert>

/* Determine if using GPU acceleration */
#ifdef GPU_BLOCK_SIZE
    #ifndef USE_CUDA
        #include "hip/hip_runtime.h"
        #define HIP_ASSERT(x) (assert((x)==hipSuccess))
    #endif
    #ifdef USE_CUDA
        #include <cuda_runtime.h>
        #define CUDA_ASSERT(x) (assert((x)==cudaSuccess))
    #endif
    #ifndef MAX_GPU_STREAMS
        #define MAX_GPU_STREAMS 1 ///< Max number of concurrent GPU streams
    #endif
#endif

#endif
