#if defined(USE_CUDA)
#include "deac_gpu.cuh"
#elif defined(USE_HIP)
#include "deac_gpu.hip.hpp"
#elif defined(USE_SYCL)
#include "deac_gpu.sycl.h"
#else
#error "gpu_normalization_test requires a GPU backend"
#endif

#include <array>
#include <cfloat>
#include <iostream>
#include <limits>

int main() {
    constexpr std::size_t rows = 4;
    constexpr std::size_t cols = 2;
    constexpr std::size_t matrix_elements = rows*cols;

    // Column-major candidate rows: the first has a valid denominator and the
    // other rows model a zero denominator, a subnormal denominator, and a
    // denominator that would produce a subnormal scale, respectively.
    std::array<double, matrix_elements> positive{
            1.0, 9.0, 8.0, 7.0, 3.0, 6.0, 5.0, 4.0};
    std::array<double, matrix_elements> negative{
            2.0, 8.0, 7.0, 6.0, 4.0, 5.0, 4.0, 3.0};
    const std::array<double, rows> denominator{
            4.0,
            0.0,
            std::numeric_limits<double>::denorm_min(),
            DBL_MAX};
    std::array<bool, rows> valid{};
    std::array<double, rows> fitness_new{1.0, 0.25, 0.5, 0.75};
    std::array<double, rows> fitness_old{2.0, 2.0, 2.0, 2.0};
    std::array<bool, rows> accepted{};

    deac_stream_t stream;
    GPU_ASSERT(deac_stream_create(stream));
    double* d_positive;
    double* d_negative;
    double* d_denominator;
    bool* d_valid;
    double* d_fitness_new;
    double* d_fitness_old;
    bool* d_accepted;
    GPU_ASSERT(deac_malloc_device(double, d_positive, matrix_elements, stream));
    GPU_ASSERT(deac_malloc_device(double, d_negative, matrix_elements, stream));
    GPU_ASSERT(deac_malloc_device(double, d_denominator, rows, stream));
    GPU_ASSERT(deac_malloc_device(bool, d_valid, rows, stream));
    GPU_ASSERT(deac_malloc_device(double, d_fitness_new, rows, stream));
    GPU_ASSERT(deac_malloc_device(double, d_fitness_old, rows, stream));
    GPU_ASSERT(deac_malloc_device(bool, d_accepted, rows, stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_positive, positive.data(), sizeof(positive), stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_negative, negative.data(), sizeof(negative), stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_denominator, denominator.data(), sizeof(denominator), stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_fitness_new, fitness_new.data(), sizeof(fitness_new), stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_fitness_old, fitness_old.data(), sizeof(fitness_old), stream));
    GPU_ASSERT(deac_wait(stream));

    gpu_deac_dgmmDiv1D(
            stream, d_positive, d_negative, d_denominator, d_valid,
            2.0, rows, cols);
    gpu_set_rejection_indices(
            stream, 1, d_accepted, d_fitness_new, d_fitness_old,
            d_valid, true, rows);
    GPU_ASSERT(deac_memcpy_device_to_host(
            positive.data(), d_positive, sizeof(positive), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            negative.data(), d_negative, sizeof(negative), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            valid.data(), d_valid, sizeof(valid), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            fitness_new.data(), d_fitness_new, sizeof(fitness_new), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            fitness_old.data(), d_fitness_old, sizeof(fitness_old), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            accepted.data(), d_accepted, sizeof(accepted), stream));
    GPU_ASSERT(deac_wait(stream));

    const bool valid_row_unchanged = valid[0]
            && positive[0] == 0.5 && positive[rows] == 1.5
            && negative[0] == 1.0 && negative[rows] == 2.0
            && accepted[0] && fitness_old[0] == 1.0;
    bool invalid_rows_rejected = true;
    for (std::size_t row=1; row<rows; ++row) {
        invalid_rows_rejected = invalid_rows_rejected
                && !valid[row]
                && positive[row] == 0.0 && positive[row + rows] == 0.0
                && negative[row] == 0.0 && negative[row + rows] == 0.0
                && !accepted[row] && fitness_new[row] == DBL_MAX
                && fitness_old[row] == 2.0;
    }

    GPU_ASSERT(deac_free(d_positive, stream));
    GPU_ASSERT(deac_free(d_negative, stream));
    GPU_ASSERT(deac_free(d_denominator, stream));
    GPU_ASSERT(deac_free(d_valid, stream));
    GPU_ASSERT(deac_free(d_fitness_new, stream));
    GPU_ASSERT(deac_free(d_fitness_old, stream));
    GPU_ASSERT(deac_free(d_accepted, stream));
    GPU_ASSERT(deac_wait(stream));
    GPU_ASSERT(deac_stream_destroy(stream));

    if (!valid_row_unchanged) {
        std::cerr << "valid GPU normalization row changed unexpectedly\n";
        return 1;
    }
    if (!invalid_rows_rejected) {
        std::cerr << "degenerate GPU normalization row was not rejected\n";
        return 1;
    }
    return 0;
}
