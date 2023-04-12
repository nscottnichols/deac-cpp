#ifndef DEAC_DEBUG_H 
#define DEAC_DEBUG_H
#ifdef USE_HIP
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
