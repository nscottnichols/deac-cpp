#ifndef COMMON_GPU_H 
#define COMMON_GPU_H
#include <cassert>

#define IS_POWER_OF_TWO(x) ((x != 0) && ((x & (x - 1)) == 0))

/* Determine if using GPU acceleration */
#ifdef USE_GPU
    #ifndef GPU_BLOCK_SIZE
        #define GPU_BLOCK_SIZE 256 ///< number of threads per block
    #endif
    
    #if GPU_BLOCK_SIZE > 2048
        #error "GPU_BLOCK_SIZE > 2048 unsupported"
    #endif

    #if !IS_POWER_OF_TWO(GPU_BLOCK_SIZE)
        #error "GPU_BLOCK_SIZE must be a power of two"
    #endif

    #ifndef MAX_GPU_STREAMS
        #define MAX_GPU_STREAMS 1 ///< Max number of concurrent GPU streams
    #endif
    #ifdef USE_HIP
        #include "hip/hip_runtime.h"
        #define GPU_ASSERT(x) (assert((x)==hipSuccess))
        typedef hipStream_t deac_stream_t;
        #define deac_stream_create(x) hipStreamCreate(&x)
        #define deac_stream_destroy(x) hipStreamDestroy(x)
        #define deac_malloc_device(T, x, y, z) hipMallocAsync(&x, sizeof(T)*y, z)
        #define deac_memcopy_host_to_device(w, x, y, z) hipMemcpyAsync(w, x, y, hipMemcpyHostToDevice, z)
        #define deac_memcopy_device_to_host(w, x, y, z) hipMemcpyAsync(w, x, y, hipMemcpyDeviceToHost, z)
        #define deac_wait(x) hipStreamSynchronize(x)
        #define deac_memset(w, x, y, z) hipMemsetAsync(w, x, y, z)
    #endif
    #ifdef USE_CUDA
        #include <cuda_runtime.h>
        #define GPU_ASSERT(x) (assert((x)==cudaSuccess))
        typedef cudaStream_t deac_stream_t;
        #define deac_stream_create(x) cudaStreamCreate(&x)
        #define deac_stream_destroy(x) cudaStreamDestroy(x)
        #define deac_malloc_device(T, x, y, z) cudaMallocAsync(&x, sizeof(T)*y, z)
        #define deac_memcopy_host_to_device(w, x, y, z) cudaMemcpyAsync(w, x, y, cudaMemcpyHostToDevice, z)
        #define deac_memcopy_device_to_host(w, x, y, z) cudaMemcpyAsync(w, x, y, cudaMemcpyDeviceToHost, z)
        #define deac_wait(x) cudaStreamSynchronize(x)
        #define deac_memset(w, x, y, z) cudaMemsetAsync(w, x, y, z)
    #endif
    #ifdef USE_SYCL
        #include <CL/sycl.hpp>
        #define GPU_ASSERT(x) x
        #ifndef SUB_GROUP_SIZE
            #define SUB_GROUP_SIZE 8 ///< number of threads per subgroup
        #endif
        #if !IS_POWER_OF_TWO(SUB_GROUP_SIZE)
            #error "SUB_GROUP_SIZE must be a power of two"
        #endif
        #if GPU_BLOCK_SIZE < 2*SUB_GROUP_SIZE
            #error "GPU_BLOCK_SIZE must be >= 2*SUB_GROUP_SIZE"
        #endif 
        typedef sycl::queue deac_stream_t;
        #define deac_stream_create(x) x = sycl::queue()
        #define deac_stream_destroy(x) do {} while(0)
        #define deac_malloc_device(T, x, y, z) x = sycl::malloc_device< T >( y, z )
        #define deac_memcopy_host_to_device(w, x, y, z) z.memcpy(w, x, y)
        #define deac_memcopy_device_to_host(w, x, y, z) z.memcpy(w, x, y)
        #define deac_wait(x) x.wait()
        #define deac_memset(w, x, y, z) z.memset(w, x, y)
    #endif
#endif
#endif
