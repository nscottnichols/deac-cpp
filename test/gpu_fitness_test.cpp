#if defined(USE_CUDA)
#include "deac_gpu.cuh"
#elif defined(USE_HIP)
#include "deac_gpu.hip.hpp"
#elif defined(USE_SYCL)
#include "deac_gpu.sycl.h"
#else
#error "gpu_fitness_test requires a GPU backend"
#endif

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {
// Keep the negative-control harness independent of production GPU_ASSERT:
// exact base 16ce erases CUDA/HIP GPU_ASSERT operations under NDEBUG, which
// would confound a proof that changes only the beta-zero kernel semantics.
#if defined(USE_CUDA)
void check_gpu_call(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        std::cerr << expression << " failed: " << cudaGetErrorString(status)
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}
#define TEST_GPU_CALL(expression) check_gpu_call((expression), #expression)
#elif defined(USE_HIP)
void check_gpu_call(hipError_t status, const char* expression) {
    if (status != hipSuccess) {
        std::cerr << expression << " failed: " << hipGetErrorString(status)
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}
#define TEST_GPU_CALL(expression) check_gpu_call((expression), #expression)
#else
// SYCL operations either throw directly or report asynchronous errors through
// the explicit deac_wait calls below.
#define TEST_GPU_CALL(expression) \
    do {                          \
        expression;              \
    } while (false)
#endif

bool check_values(
        const char* label,
        const std::vector<double>& actual,
        const std::vector<double>& expected,
        std::size_t dimension) {
    for (std::size_t row=0; row<dimension; ++row) {
        if (!std::isfinite(actual[row]) || actual[row] != expected[row]) {
            std::cerr << label << " mismatch at dimension " << dimension
                      << ", row " << row << ": expected " << std::hexfloat
                      << expected[row] << ", got " << actual[row] << '\n';
            return false;
        }
    }
    return true;
}

bool check_dimension(deac_stream_t stream, std::size_t dimension) {
    const std::size_t matrix_elements = dimension*dimension;
    std::vector<double> calculated(matrix_elements);
    std::vector<double> observed(dimension, 8.0);
    std::vector<double> standard_deviations(dimension, 2.0);
    std::vector<double> scalar_calculated(dimension);
    std::vector<double> base_expected(dimension);
    std::vector<double> scalar_expected(dimension);
    std::vector<double> nonzero_beta_expected(dimension);
    std::vector<double> sentinels(dimension);
    std::vector<double> fitness(
            dimension, std::numeric_limits<double>::quiet_NaN());

    // Each row has one exact integer residual across every column.  Its mean
    // square is therefore residual^2 regardless of the reduction tree.
    for (std::size_t row=0; row<dimension; ++row) {
        const double residual = static_cast<double>(row%3 + 1);
        base_expected[row] = residual*residual;
        scalar_expected[row] = 5.0*base_expected[row];
        sentinels[row] = 0.25*static_cast<double>(row%4 + 1);
        nonzero_beta_expected[row] =
                base_expected[row] + 2.0*sentinels[row];
        scalar_calculated[row] =
                observed[0] - standard_deviations[0]*residual;
        for (std::size_t column=0; column<dimension; ++column) {
            calculated[row + dimension*column] =
                    observed[column]
                    - standard_deviations[column]*residual;
        }
    }

    double* d_calculated = nullptr;
    double* d_observed = nullptr;
    double* d_standard_deviations = nullptr;
    double* d_scalar_calculated = nullptr;
    double* d_fitness = nullptr;
    TEST_GPU_CALL(deac_malloc_device(
            double, d_calculated, matrix_elements, stream));
    TEST_GPU_CALL(deac_malloc_device(
            double, d_observed, dimension, stream));
    TEST_GPU_CALL(deac_malloc_device(
            double, d_standard_deviations, dimension, stream));
    TEST_GPU_CALL(deac_malloc_device(
            double, d_scalar_calculated, dimension, stream));
    TEST_GPU_CALL(deac_malloc_device(
            double, d_fitness, dimension, stream));
    TEST_GPU_CALL(deac_memcpy_host_to_device(
            d_calculated, calculated.data(),
            matrix_elements*sizeof(double), stream));
    TEST_GPU_CALL(deac_memcpy_host_to_device(
            d_observed, observed.data(), dimension*sizeof(double), stream));
    TEST_GPU_CALL(deac_memcpy_host_to_device(
            d_standard_deviations, standard_deviations.data(),
            dimension*sizeof(double), stream));
    TEST_GPU_CALL(deac_memcpy_host_to_device(
            d_scalar_calculated, scalar_calculated.data(),
            dimension*sizeof(double), stream));
    TEST_GPU_CALL(deac_wait(stream));

    // A zero beta must overwrite every poisoned destination without reading it.
    TEST_GPU_CALL(deac_memcpy_host_to_device(
            d_fitness, fitness.data(), dimension*sizeof(double), stream));
    TEST_GPU_CALL(deac_wait(stream));
    gpu_deac_reduced_chi_squared(
            stream, d_calculated, d_observed, d_standard_deviations,
            d_fitness, dimension, dimension, 0, 0.0);
    TEST_GPU_CALL(deac_wait(stream));
    TEST_GPU_CALL(deac_memcpy_device_to_host(
            fitness.data(), d_fitness, dimension*sizeof(double), stream));
    TEST_GPU_CALL(deac_wait(stream));
    bool valid = check_values(
            "zero-beta reduced chi squared",
            fitness, base_expected, dimension);

    // The next production step must add its scalar penalty to the overwritten
    // values, including rows in a partial block beyond the first block.
    gpu_deac_add_scalar_reduced_chi_squared(
            stream, d_scalar_calculated, observed[0], 1.0,
            d_fitness, dimension);
    TEST_GPU_CALL(deac_wait(stream));
    TEST_GPU_CALL(deac_memcpy_device_to_host(
            fitness.data(), d_fitness, dimension*sizeof(double), stream));
    TEST_GPU_CALL(deac_wait(stream));
    valid = check_values(
            "scalar reduced chi squared",
            fitness, scalar_expected, dimension) && valid;

    // Preserve the existing nonzero-beta arithmetic and destination read.
    TEST_GPU_CALL(deac_memcpy_host_to_device(
            d_fitness, sentinels.data(), dimension*sizeof(double), stream));
    TEST_GPU_CALL(deac_wait(stream));
    gpu_deac_reduced_chi_squared(
            stream, d_calculated, d_observed, d_standard_deviations,
            d_fitness, dimension, dimension, 0, 2.0);
    TEST_GPU_CALL(deac_wait(stream));
    TEST_GPU_CALL(deac_memcpy_device_to_host(
            fitness.data(), d_fitness, dimension*sizeof(double), stream));
    TEST_GPU_CALL(deac_wait(stream));
    valid = check_values(
            "nonzero-beta reduced chi squared",
            fitness, nonzero_beta_expected, dimension) && valid;

    TEST_GPU_CALL(deac_free(d_calculated, stream));
    TEST_GPU_CALL(deac_free(d_observed, stream));
    TEST_GPU_CALL(deac_free(d_standard_deviations, stream));
    TEST_GPU_CALL(deac_free(d_scalar_calculated, stream));
    TEST_GPU_CALL(deac_free(d_fitness, stream));
    TEST_GPU_CALL(deac_wait(stream));
    return valid;
}
} // namespace

int main() {
    deac_stream_t stream;
    TEST_GPU_CALL(deac_stream_create(stream));

    const std::size_t below_block =
            GPU_BLOCK_SIZE > 1 ? GPU_BLOCK_SIZE - 1 : 1;
    const std::size_t above_block = GPU_BLOCK_SIZE + 17;
    const bool below_valid = check_dimension(stream, below_block);
    const bool above_valid = check_dimension(stream, above_block);

    TEST_GPU_CALL(deac_stream_destroy(stream));
    return below_valid && above_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
