/**
 * @file deac_gpu_sycl.h
 * @author Nathan Nichols
 * @date 04.12/2023
 *
 * @brief GPU kernels using SYCL.
 */

#ifndef DEAC_GPU_SYCL_H 
#define DEAC_GPU_SYCL_H
#include "common_gpu.hpp"

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GPU KERNELS ---------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

uint64_t gpu_rol64(uint64_t x, uint64_t k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t gpu_xoshiro256p_next(uint64_t* s) {
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

void sub_group_reduce_add(volatile double* _c, size_t local_idx) {
    #if (SUB_GROUP_SIZE >= 512)
        _c[local_idx] += _c[local_idx + 512];
    #endif
    #if (SUB_GROUP_SIZE >= 256)
        _c[local_idx] += _c[local_idx + 256];
    #endif
    #if (SUB_GROUP_SIZE >= 128)
        _c[local_idx] += _c[local_idx + 128];
    #endif
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

template <int Dimensions = 1>
void gpu_reduce_add(double* _c, sycl::nd_item<Dimensions> item) {
    size_t local_idx = item.get_local_id(0);
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        if (local_idx < 512) {
            _c[local_idx] += _c[local_idx + 512];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        if (local_idx < 256) {
            _c[local_idx] += _c[local_idx + 256];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        if (local_idx < 128) {
            _c[local_idx] += _c[local_idx + 128];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        if (local_idx < 64) {
            _c[local_idx] += _c[local_idx + 64];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
        if (local_idx < 32) {
            _c[local_idx] += _c[local_idx + 32];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
        if (local_idx < 16) {
            _c[local_idx] += _c[local_idx + 16];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
        if (local_idx < 8) {
            _c[local_idx] += _c[local_idx + 8];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
        if (local_idx < 4) {
            _c[local_idx] += _c[local_idx + 4];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
        if (local_idx < 2) {
            _c[local_idx] += _c[local_idx + 2];
        }
        sycl::group_barrier(item.get_group());
    #endif

    //Sub-group reduce
    if (local_idx < SUB_GROUP_SIZE) {
        sub_group_reduce_add(_c, local_idx);
    }
    sycl::group_barrier(item.get_group());
}

void sub_group_reduce_min(volatile double* _c, size_t local_idx) {
    #if (SUB_GROUP_SIZE >= 512)
        _c[local_idx] = _c[local_idx + 512] < _c[local_idx] ? _c[local_idx + 512] : _c[local_idx];
    #endif
    #if (SUB_GROUP_SIZE >= 256)
        _c[local_idx] = _c[local_idx + 256] < _c[local_idx] ? _c[local_idx + 256] : _c[local_idx];
    #endif
    #if (SUB_GROUP_SIZE >= 128)
        _c[local_idx] = _c[local_idx + 128] < _c[local_idx] ? _c[local_idx + 128] : _c[local_idx];
    #endif
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

template <int Dimensions = 1>
void gpu_reduce_min(double* _c, sycl::nd_item<Dimensions> item) {
    size_t local_idx = item.get_local_id(0);
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        if (local_idx < 512) {
            _c[local_idx] = _c[local_idx + 512] < _c[local_idx] ? _c[local_idx + 512] : _c[local_idx];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        if (local_idx < 256) {
            _c[local_idx] = _c[local_idx + 256] < _c[local_idx] ? _c[local_idx + 256] : _c[local_idx];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        if (local_idx < 128) {
            _c[local_idx] = _c[local_idx + 128] < _c[local_idx] ? _c[local_idx + 128] : _c[local_idx];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        if (local_idx < 64) {
            _c[local_idx] = _c[local_idx + 64] < _c[local_idx] ? _c[local_idx + 64] : _c[local_idx];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
        if (local_idx < 32) {
            _c[local_idx] = _c[local_idx + 32] < _c[local_idx] ? _c[local_idx + 32] : _c[local_idx];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
        if (local_idx < 16) {
            _c[local_idx] = _c[local_idx + 16] < _c[local_idx] ? _c[local_idx + 16] : _c[local_idx];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
        if (local_idx < 8) {
            _c[local_idx] = _c[local_idx + 8] < _c[local_idx] ? _c[local_idx + 8] : _c[local_idx];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
        if (local_idx < 4) {
            _c[local_idx] = _c[local_idx + 4] < _c[local_idx] ? _c[local_idx + 4] : _c[local_idx];
        }
        sycl::group_barrier(item.get_group());
    #endif

    #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
        if (local_idx < 2) {
            _c[local_idx] = _c[local_idx + 2] < _c[local_idx] ? _c[local_idx + 2] : _c[local_idx];
        }
        sycl::group_barrier(item.get_group());
    #endif

    //Sub-group reduce
    if (local_idx < SUB_GROUP_SIZE) {
        sub_group_reduce_min(_c, local_idx);
    }
    sycl::group_barrier(item.get_group());
}

void gpu_dot(sycl::queue q, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
    // C = B*A where [B] = 1xN and [A] = Nx1
    q.submit([&](sycl::handler& cgh) {
        // Shared Local Memory _c
        sycl::local_accessor<double, 1> _c(sycl::range<1>(GPU_BLOCK_SIZE), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            // Set shared local memory _c
            size_t local_idx = item.get_local_id(0);
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
            sycl::group_barrier(item.get_group());

            // Reduce _c (using shared local memory)
            gpu_reduce_add(_c.get_pointer(), item);

            //Set C
            if (local_idx == 0) {
                 C[0] += _c[0]; //FIXME should do C[0] = _c[0] + scale_factor*C[0] here probably
            }
        });
    });
}

void gpu_matmul_simple(sycl::queue q, int m, int n, int k, double alpha, double* __restrict__ A, int lda, double* __restrict__ B, int ldb, double beta, double* __restrict__ C, int ldc) {
    q.submit([&](sycl::handler& cgh) {
        size_t grid_size_x = (n + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
        size_t grid_size_y = (m + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(grid_size_x*GPU_BLOCK_SIZE, grid_size_y*GPU_BLOCK_SIZE), sycl::range<2>(GPU_BLOCK_SIZE, GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            int row = item.get_group_id(1) * item.get_local_range(1) + item.get_local_id(1);
            int col = item.get_group_id(0) * item.get_local_range(0) + item.get_local_id(0);

            if (row < m && col < n) {
                double sum = 0.0;
                for (int e = 0; e < k; e++) {
                    sum += A[row + e * lda] * B[e + col * ldb];
                }
                C[row + col * ldc] = alpha * sum + beta * C[row + col * ldc];
            }
        });
    });
}

void gpu_matmul(sycl::queue q, int m, int n, int k, double alpha, double* __restrict__ A, int lda, double* __restrict__ B, int ldb, double beta, double* __restrict__ C, int ldc) {
    q.submit([&](sycl::handler& cgh) {
        size_t grid_size_x = (n + TILE_WIDTH - 1) / TILE_WIDTH;
        size_t grid_size_y = (m + TILE_WIDTH - 1) / TILE_WIDTH;
        sycl::local_accessor<double, 2> As(sycl::range<2>(TILE_WIDTH, TILE_WIDTH), cgh);
        sycl::local_accessor<double, 2> Bs(sycl::range<2>(TILE_WIDTH, TILE_WIDTH), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(grid_size_x*TILE_WIDTH, grid_size_y*TILE_WIDTH), sycl::range<2>(TILE_WIDTH, TILE_WIDTH)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            int bx = item.get_group_id(0), by = item.get_group_id(1);
            int tx = item.get_local_id(0), ty = item.get_local_id(1);

            // Identify the row and column of the C element to work on
            int row = by * TILE_WIDTH + ty;
            int col = bx * TILE_WIDTH + tx;

            double Cvalue = 0.0;

            // Loop over the A and B tiles required to compute the C element
            for (int t = 0; t < (k-1)/TILE_WIDTH + 1; ++t) {

                // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
                if (row < m && t*TILE_WIDTH+tx < k)
                    As[ty][tx] = A[row + lda * (t*TILE_WIDTH+tx)];
                else
                    As[ty][tx] = 0.0;

                if (t*TILE_WIDTH+ty < k && col < n)
                    Bs[ty][tx] = B[(t*TILE_WIDTH+ty) + ldb * col];
                else
                    Bs[ty][tx] = 0.0;

                sycl::group_barrier(item.get_group()); // Make sure the matrices are loaded before starting the computation

                // Multiply the two matrices together; each thread computes one element of the block sub-matrix
                for (int e = 0; e < TILE_WIDTH; ++e) {
                    Cvalue += As[ty][e] * Bs[e][tx];
                }

                sycl::group_barrier(item.get_group()); // Make sure that all threads are done computing before loading the next set of tiles
            }

            if (row < m && col < n)
                C[row + ldc * col] = alpha * Cvalue + beta * C[row + ldc * col];
        });
    });
}

void gpu_deac_gemv_simple(sycl::queue q, int m, int n, double alpha, double* __restrict__ A, int lda, double* __restrict__ x, int incx, double beta, double* __restrict__ y, int incy) {
    q.submit([&](sycl::handler& cgh) {
        size_t grid_size = (m + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            int row = item.get_group_id(0) * item.get_local_range(0) + item.get_local_id(0);
            if (row < m) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += A[row + j*lda] * x[j*incx];
                }
                y[row*incy] = alpha * sum + beta * y[row*incy];
            }
        });
    });
}

void gpu_deac_gemv_atomic(sycl::queue q, int m, int n, double alpha, double* __restrict__ A, int lda, double* __restrict__ x, int incx, double beta, double* __restrict__ y, int incy) {
    q.submit([&](sycl::handler& cgh) {
        size_t grid_size = (n + TILE_WIDTH - 1) / TILE_WIDTH;
        sycl::local_accessor<double, 1> shared_x(sycl::range<1>(TILE_WIDTH), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*TILE_WIDTH), sycl::range<1>(TILE_WIDTH)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            int col = item.get_group_id(0) * item.get_local_range(0) + item.get_local_id(0);

            if (col < n) {
                shared_x[item.get_local_id(0)] = x[col * incx];
            }
            sycl::group_barrier(item.get_group());

            if (col < n) {
                for (int i = 0; i < m; i++) {
                    double Aval = A[i + col * lda];
                    sycl::atomic_ref atomicY(y[i * incy]);
                    atomicY.fetch_add(alpha * Aval * shared_x[item.get_local_id(0)]);
                }
            }
        });
    });
}

void gpu_deac_gemv(sycl::queue q, int m, int n, double alpha, double* __restrict__ A, int lda, double* __restrict__ x, int incx, double beta, double* __restrict__ y, int incy) {
    q.submit([&](sycl::handler& cgh) {
        size_t grid_size_x = (n + TILE_WIDTH - 1) / TILE_WIDTH;
        size_t grid_size_y = (m + TILE_WIDTH - 1) / TILE_WIDTH;
        sycl::local_accessor<double, 2> As(sycl::range<2>(TILE_WIDTH, TILE_WIDTH), cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(grid_size_x*TILE_WIDTH, grid_size_y*TILE_WIDTH), sycl::range<2>(TILE_WIDTH, TILE_WIDTH)),
                [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            int tx = item.get_local_id(0);
            int by = item.get_group_id(1), ty = item.get_local_id(1);
            int row = by * item.get_local_range(1) + ty;

            double sum = 0.0;
            if (row < m) {
                for (int i = 0; i < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++i) {
                    if (i*TILE_WIDTH + tx < n && row < m) {
                        As[ty][tx] = A[row + (i*TILE_WIDTH + tx) * lda];
                    } else {
                        As[ty][tx] = 0.0;
                    }
                    sycl::group_barrier(item.get_group());

                    for (int k = 0; k < TILE_WIDTH; ++k) {
                        if (i*TILE_WIDTH + k < n) {
                            sum += As[ty][k] * x[(i*TILE_WIDTH + k)*incx];
                        }
                    }
                    sycl::group_barrier(item.get_group());
                }
                if (beta == 0.0) {
                    y[row * incy] = alpha * sum;
                } else {
                    y[row * incy] = alpha * sum + beta * y[row * incy];
                }
            }
        });
    });
}

void gpu_get_minimum(sycl::queue q, double* __restrict__ minimum, double* __restrict__ array, size_t N) {
    // finds minimum of array with length N
    q.submit([&](sycl::handler& cgh) {
        // Shared Local Memory _c
        sycl::local_accessor<double, 1> _c(sycl::range<1>(GPU_BLOCK_SIZE), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            // Set shared local memory _c
            size_t local_idx = item.get_local_id(0);
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
            sycl::group_barrier(item.get_group());

            // Reduce _c (using shared local memory)
            gpu_reduce_min(_c.get_pointer(), item);

            //Set minimum
            if (local_idx == 0) {
                 minimum[0] = _c[0];
            }
        });
    });
}

void gpu_deac_dgmmDiv1D(sycl::queue q, double* __restrict__ matrix, double* __restrict__ vector, size_t rows, size_t cols) {
    q.submit([&](sycl::handler& cgh) {
        size_t grid_size = (rows*cols + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t idx = item.get_global_id(0); // 1D index for the entire matrix
            if (idx < rows * cols) { // Ensure we do not go out of bounds
                size_t row = idx % rows; // Calculate the row index
                //size_t col = idx / rows; // Calculate the column index, assuming column-major order
                // Perform the division operation
                double reciprocal = 1.0/vector[row];
                matrix[idx] *= reciprocal;
            }
        });
    });
}

void gpu_deac_reduced_chi_squared(const double* __restrict__ calculated_data, const double* __restrict__ observed_data, const double* __restrict__ standard_deviations, double* __restrict__ reduced_chi_squared, size_t m, size_t n, size_t ddof, double beta) {
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 1> sdata(sycl::range<1>(GPU_BLOCK_SIZE), cgh); // Use static shared memory for reduction
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(m*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t row = item.get_group(0);
            size_t tid = item.get_local_id(0);
            double sum = 0.0;

            // Loop over all elements assigned to this thread
            for (size_t idx = tid; idx < n; idx += item.get_local_range(0)) {
                double O = observed_data[idx];
                double E = calculated_data[row + idx * m]; // Access pattern for column-major storage
                double sigma = standard_deviations[idx];
                double term = (O - E) / sigma;
                sum += term * term;
            }

            // Load the thread's sum into shared memory and synchronize
            sdata[tid] = sum;
            sycl::group_barrier(item.get_group());

            // Perform reduction in shared memory
            for (size_t s = item.get_local_range(0) / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    sdata[tid] += sdata[tid + s];
                }
                sycl::group_barrier(item.get_group());
            }

            // Have the first thread in the block write the result for this row to global memory
            if (tid == 0) {
                reduced_chi_squared[row] = sdata[0]/(n - ddof) + beta*reduced_chi_squared[row];
            }
        });
    });
}

void gpu_set_fitness_mean(sycl::queue q, double* __restrict__ fitness_mean, double* __restrict__ fitness, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 1> _fm(sycl::range<1>(GPU_BLOCK_SIZE), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            // Set shared local memory _fm
            size_t local_idx = item.get_local_id(0);
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
            sycl::group_barrier(item.get_group());
            
            // Reduce _fm (using shared local memory)
            gpu_reduce_add(_fm.get_pointer(), item);

            //Set fitness_mean
            if (local_idx == 0) {
                 fitness_mean[0] += _fm[0]/population_size;
            }
        });
    });
}

void gpu_set_fitness_squared_mean(sycl::queue q, double* __restrict__ fitness_squared_mean, double* __restrict__ fitness, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 1> _fsm(sycl::range<1>(GPU_BLOCK_SIZE), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            // Set shared local memory _fsm
            size_t local_idx = item.get_local_id(0);
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
            sycl::group_barrier(item.get_group());
            
            // Reduce _fsm (using shared local memory)
            gpu_reduce_add(_fsm.get_pointer(), item);

            //Set fitness_squared_mean
            if (local_idx == 0) {
                 fitness_squared_mean[0] += _fsm[0]/population_size;
            }
        });
    });
}

void gpu_set_population_new(sycl::queue q, size_t grid_size, double* __restrict__ population_new, double* __restrict__ population_old, size_t* __restrict__ mutant_indices, double* __restrict__ differential_weights_new, bool* __restrict__ mutate_indices, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size*genome_size) {
                //size_t _i = global_idx/genome_size;
                //size_t _j = global_idx - _i*genome_size;
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
                        //population_new[global_idx] = sycl::fabs(population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]));
                        population_new[global_idx] = sycl::fabs(population_old[population_size*_j + mutant_index1] + F*(population_old[population_size*_j + mutant_index2] - population_old[population_size*_j + mutant_index3]));
                    #endif
                } else {
                    population_new[global_idx] = population_old[global_idx];
                }
            }
        });
    });
}

void gpu_match_population_zero(sycl::queue q, size_t grid_size, double* __restrict__ population_negative_frequency, double* __restrict__ population_positive_frequency, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size) {
                //population_negative_frequency[global_idx*genome_size] = population_positive_frequency[global_idx*genome_size];
                population_negative_frequency[global_idx] = population_positive_frequency[global_idx]; // Column-major
            }
        });
    });
}

void gpu_set_rejection_indices(sycl::queue q, size_t grid_size, bool* __restrict__ rejection_indices, double* __restrict__ fitness_new, double* __restrict__ fitness_old, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size) {
                bool accept = fitness_new[global_idx] <= fitness_old[global_idx];
                rejection_indices[global_idx] = accept;
                if (accept) {
                    fitness_old[global_idx] = fitness_new[global_idx];
                }
            }
        });
    });
}

void gpu_swap_control_parameters(sycl::queue q, size_t grid_size, double* __restrict__ control_parameter_old, double* __restrict__ control_parameter_new, bool* __restrict__ rejection_indices, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size) {
                if (rejection_indices[global_idx]) {
                    control_parameter_old[global_idx] = control_parameter_new[global_idx];
                }
            }
        });
    });
}

void gpu_swap_populations(sycl::queue q, size_t grid_size, double* __restrict__ population_old, double* __restrict__ population_new, bool* __restrict__ rejection_indices, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size*genome_size) {
                size_t _i = global_idx/genome_size;
                if (rejection_indices[_i]) {
                    population_old[global_idx] = population_new[global_idx];
                }
            }
        });
    });
}

void gpu_set_crossover_probabilities_new(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ crossover_probabilities_new, double* __restrict__ crossover_probabilities_old, double self_adapting_crossover_probability, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size) {
                if ((gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
                    crossover_probabilities_new[global_idx] = (gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53;
                } else {
                    crossover_probabilities_new[global_idx] = crossover_probabilities_old[global_idx];
                }
            }
        });
    });
}

void gpu_set_differential_weights_new(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ differential_weights_new, double* __restrict__ differential_weights_old, double self_adapting_differential_weight_probability, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size) {
                if ((gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
                    differential_weights_new[global_idx] = 2.0*((gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53);
                } else {
                    differential_weights_new[global_idx] = differential_weights_old[global_idx];
                }
            }
        });
    });
}

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

void gpu_set_mutant_indices(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, size_t* __restrict__ mutant_indices, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size) {
                gpu_set_mutant_indices(rng_state + 4*global_idx, mutant_indices + 3*global_idx, global_idx, population_size);
            }
        });
    });
}

void gpu_set_mutate_indices(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, bool* __restrict__ mutate_indices, double* __restrict__ crossover_probabilities, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size*genome_size) {
                //size_t _i = global_idx/genome_size;
                size_t _i = global_idx % population_size;
                mutate_indices[global_idx] = (gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < crossover_probabilities[_i];
            }
        });
    });
}

#ifdef USE_BLAS
    void gpu_blas_gemv(sycl::queue q, int m, int n, double alpha, double* A, double* B, double beta, double* C) {
        oneapi::mkl::blas::column_major::gemv(q, oneapi::mkl::transpose::nontrans, m, n, &alpha, A, m, B, 1, &beta, C, 1);
    }

    void gpu_blas_gemm(hipblasHandle_t handle, int m, int n, int k, double alpha, double* A, double* B, double beta, double* C) {
        oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    }
#endif
#endif
