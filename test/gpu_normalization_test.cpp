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
#include <vector>

int main() {
    constexpr std::size_t rows = 5;
    // Span more than one device block so every element-parallel normalization
    // pass is exercised beyond a single launch block.
    constexpr std::size_t cols = GPU_BLOCK_SIZE + 17;
    constexpr std::size_t matrix_elements = rows*cols;

    // Column-major candidate rows: the first has a valid denominator and the
    // other rows model a zero denominator, a subnormal denominator, a
    // denominator that would produce a subnormal scale, and one isolated
    // overflowing matrix product, respectively.
    std::vector<double> positive(matrix_elements);
    std::vector<double> negative(matrix_elements);
    for (std::size_t col=0; col<cols; ++col) {
        for (std::size_t row=0; row<rows; ++row) {
            const std::size_t index = row + rows*col;
            positive[index] = static_cast<double>(1 + col%7 + row);
            negative[index] = static_cast<double>(2 + col%5 + row);
        }
    }
    // These values distinguish the legacy x*(1/scaled_denominator) sequence
    // from regrouping it as x*(target/(scaled_denominator*target)): the latter
    // rounds the first positive result one ulp below 1.0.
    positive[0] = 0.1;
    negative[0] = 0.2;
    positive[4] = DBL_MAX;
    const std::array<double, rows> denominator{
            0.1,
            0.0,
            std::numeric_limits<double>::denorm_min(),
            DBL_MAX/2.0,
            0.5};
    std::array<int, rows> valid{};
    std::array<double, rows> fitness_new{1.0, 0.25, 0.5, 0.75, 1.25};
    std::array<double, rows> fitness_old{2.0, 2.0, 2.0, 2.0, 2.0};
    std::array<bool, rows> accepted{};

    deac_stream_t stream;
    GPU_ASSERT(deac_stream_create(stream));
    double* d_positive;
    double* d_negative;
    double* d_denominator;
    int* d_valid;
    double* d_fitness_new;
    double* d_fitness_old;
    bool* d_accepted;
    GPU_ASSERT(deac_malloc_device(double, d_positive, matrix_elements, stream));
    GPU_ASSERT(deac_malloc_device(double, d_negative, matrix_elements, stream));
    GPU_ASSERT(deac_malloc_device(double, d_denominator, rows, stream));
    GPU_ASSERT(deac_malloc_device(int, d_valid, rows, stream));
    GPU_ASSERT(deac_malloc_device(double, d_fitness_new, rows, stream));
    GPU_ASSERT(deac_malloc_device(double, d_fitness_old, rows, stream));
    GPU_ASSERT(deac_malloc_device(bool, d_accepted, rows, stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_positive, positive.data(), matrix_elements*sizeof(double), stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_negative, negative.data(), matrix_elements*sizeof(double), stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_denominator, denominator.data(), sizeof(denominator), stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_fitness_new, fitness_new.data(), sizeof(fitness_new), stream));
    GPU_ASSERT(deac_memcpy_host_to_device(
            d_fitness_old, fitness_old.data(), sizeof(fitness_old), stream));
    GPU_ASSERT(deac_wait(stream));

    GPU_ASSERT(deac_memset(d_valid, 1, sizeof(valid), stream));
    gpu_deac_dgmmDiv1D(
            stream, d_positive, d_denominator, rows, cols);
    gpu_deac_dgmmDiv1D(
            stream, d_negative, d_denominator, rows, cols);
    GPU_ASSERT(deac_wait(stream));
    gpu_validate_normalization_rows(
            stream, d_positive, d_denominator, d_valid, 0.1, rows, cols);
    gpu_validate_normalization_rows(
            stream, d_negative, d_denominator, d_valid, 0.1, rows, cols);
    GPU_ASSERT(deac_wait(stream));
    gpu_cleanup_invalid_normalization_rows(
            stream, d_positive, d_valid, rows, cols);
    gpu_cleanup_invalid_normalization_rows(
            stream, d_negative, d_valid, rows, cols);
    GPU_ASSERT(deac_wait(stream));
    gpu_set_rejection_indices(
            stream, 1, d_accepted, d_fitness_new, d_fitness_old,
            d_valid, true, rows);
    GPU_ASSERT(deac_memcpy_device_to_host(
            positive.data(), d_positive, matrix_elements*sizeof(double), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            negative.data(), d_negative, matrix_elements*sizeof(double), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            valid.data(), d_valid, sizeof(valid), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            fitness_new.data(), d_fitness_new, sizeof(fitness_new), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            fitness_old.data(), d_fitness_old, sizeof(fitness_old), stream));
    GPU_ASSERT(deac_memcpy_device_to_host(
            accepted.data(), d_accepted, sizeof(accepted), stream));
    GPU_ASSERT(deac_wait(stream));

    bool valid_row_unchanged = valid[0] != 0
            && accepted[0] && fitness_old[0] == 1.0;
    bool invalid_rows_rejected = true;
    for (std::size_t col=0; col<cols; ++col) {
        const double expected_positive = col == 0
                ? 1.0 : static_cast<double>(1 + col%7)*10.0;
        const double expected_negative = col == 0
                ? 2.0 : static_cast<double>(2 + col%5)*10.0;
        valid_row_unchanged = valid_row_unchanged
                && positive[rows*col] == expected_positive
                && negative[rows*col] == expected_negative;
        for (std::size_t row=1; row<rows; ++row) {
            invalid_rows_rejected = invalid_rows_rejected
                    && !valid[row]
                    && positive[row + rows*col] == 0.0
                    && negative[row + rows*col] == 0.0
                    && !accepted[row] && fitness_new[row] == DBL_MAX
                    && fitness_old[row] == 2.0;
        }
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
