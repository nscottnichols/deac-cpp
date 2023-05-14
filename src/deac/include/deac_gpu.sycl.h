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
void gpu_reduce_add(double* _c, sycl::group<Dimensions> work_group) {
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 512) {
                _c[local_idx] += _c[local_idx + 512];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 256) {
                _c[local_idx] += _c[local_idx + 256];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 128) {
                _c[local_idx] += _c[local_idx + 128];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 64) {
                _c[local_idx] += _c[local_idx + 64];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 32) {
                _c[local_idx] += _c[local_idx + 32];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 16) {
                _c[local_idx] += _c[local_idx + 16];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 8) {
                _c[local_idx] += _c[local_idx + 8];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 4) {
                _c[local_idx] += _c[local_idx + 4];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 2) {
                _c[local_idx] += _c[local_idx + 2];
            }
        });
    #endif

    //Sub-group reduce
    work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
        size_t local_idx = index.get_local_id(0);
        if (local_idx < SUB_GROUP_SIZE) {
            sub_group_reduce_add(_c, local_idx);
        }
    });
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
void gpu_reduce_min(double* _c, sycl::group<Dimensions> work_group) {
    #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 512) {
                _c[local_idx] = _c[local_idx + 512] < _c[local_idx] ? _c[local_idx + 512] : _c[local_idx];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 256) {
                _c[local_idx] = _c[local_idx + 256] < _c[local_idx] ? _c[local_idx + 256] : _c[local_idx];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 128) {
                _c[local_idx] = _c[local_idx + 128] < _c[local_idx] ? _c[local_idx + 128] : _c[local_idx];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 64) {
                _c[local_idx] = _c[local_idx + 64] < _c[local_idx] ? _c[local_idx + 64] : _c[local_idx];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 32) {
                _c[local_idx] = _c[local_idx + 32] < _c[local_idx] ? _c[local_idx + 32] : _c[local_idx];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 16) {
                _c[local_idx] = _c[local_idx + 16] < _c[local_idx] ? _c[local_idx + 16] : _c[local_idx];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 8) {
                _c[local_idx] = _c[local_idx + 8] < _c[local_idx] ? _c[local_idx + 8] : _c[local_idx];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 4) {
                _c[local_idx] = _c[local_idx + 4] < _c[local_idx] ? _c[local_idx + 4] : _c[local_idx];
            }
        });
    #endif

    #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
        work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
            size_t local_idx = index.get_local_id(0);
            if (local_idx < 2) {
                _c[local_idx] = _c[local_idx + 2] < _c[local_idx] ? _c[local_idx + 2] : _c[local_idx];
            }
        });
    #endif

    //Sub-group reduce
    work_group.parallel_for_work_item([&](sycl::h_item<1> index) {
        size_t local_idx = index.get_local_id(0);
        if (local_idx < SUB_GROUP_SIZE) {
            sub_group_reduce_min(_c, local_idx);
        }
    });
}

void gpu_matmul(sycl::queue q, size_t grid_size, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t N) {
    // C = B*A where [B] = 1xN and [A] = Nx1
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            double _c[GPU_BLOCK_SIZE];
            // Set shared local memory _c
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < N) {
                    _c[local_idx] = A[global_idx]*B[global_idx];
                } else {
                    _c[local_idx] = 0.0;
                }
            });

            // Reduce _c (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _c[local_idx] += _c[local_idx + 512];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _c[local_idx] += _c[local_idx + 256];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _c[local_idx] += _c[local_idx + 128];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _c[local_idx] += _c[local_idx + 64];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _c[local_idx] += _c[local_idx + 32];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _c[local_idx] += _c[local_idx + 16];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _c[local_idx] += _c[local_idx + 8];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _c[local_idx] += _c[local_idx + 4];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _c[local_idx] += _c[local_idx + 2];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_c, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_c, wGroup);
            #endif

            //Set C
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t group_idx = wGroup.get_group_id(0);
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     //C_tmp[group_idx] = _c[0];
                     C[0] = _c[0];
                }
            });
        }));
    });
}

void gpu_get_minimum(sycl::queue q, double* __restrict__ minimum, double* __restrict__ array, size_t N) {
    // finds minimum of array with length N
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{1}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            double _c[GPU_BLOCK_SIZE];
            // Set shared local memory _c
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < N) {
                    _c[local_idx] = array[local_idx];
                } else {
                    _c[local_idx] = array[0];
                }

                for (size_t i = 0; i < N/GPU_BLOCK_SIZE; i++) {
                    size_t j = GPU_BLOCK_SIZE*i + local_idx;
                    if (j < N) {
                        _c[local_idx] = array[j] < _c[local_idx] ? array[j] : _c[local_idx];
                    }
                }
            });

            // Reduce _c (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _c[local_idx] = _c[local_idx + 512] < _c[local_idx] ? _c[local_idx + 512] : _c[local_idx];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _c[local_idx] = _c[local_idx + 256] < _c[local_idx] ? _c[local_idx + 256] : _c[local_idx];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _c[local_idx] = _c[local_idx + 128] < _c[local_idx] ? _c[local_idx + 128] : _c[local_idx];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _c[local_idx] = _c[local_idx + 64] < _c[local_idx] ? _c[local_idx + 64] : _c[local_idx];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _c[local_idx] = _c[local_idx + 32] < _c[local_idx] ? _c[local_idx + 32] : _c[local_idx];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _c[local_idx] = _c[local_idx + 16] < _c[local_idx] ? _c[local_idx + 16] : _c[local_idx];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _c[local_idx] = _c[local_idx + 8] < _c[local_idx] ? _c[local_idx + 8] : _c[local_idx];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _c[local_idx] = _c[local_idx + 4] < _c[local_idx] ? _c[local_idx + 4] : _c[local_idx];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _c[local_idx] = _c[local_idx + 2] < _c[local_idx] ? _c[local_idx + 2] : _c[local_idx];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_min(_c, local_idx);
                    }
                });
            #else
                gpu_reduce_min(_c, wGroup);
            #endif

            //Set C
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     minimum[0] = _c[0];
                }
            });
        }));
    });
}

void gpu_matrix_multiply_MxN_by_Nx1(sycl::queue q, size_t grid_size, double* __restrict__ C_tmp, double* __restrict__ C, double* __restrict__ B, double* __restrict__ A, size_t M, size_t N) {
    // C = B*A where [B] = MxN and [A] = Nx1
    auto event_reduce_to_C_tmp = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            double _c[GPU_BLOCK_SIZE];
            // Set shared local memory _c
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < N) {
                    _c[local_idx] = A[global_idx]*B[global_idx];
                } else {
                    _c[local_idx] = 0.0;
                }
            });

            // Reduce _c (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _c[local_idx] += _c[local_idx + 512];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _c[local_idx] += _c[local_idx + 256];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _c[local_idx] += _c[local_idx + 128];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _c[local_idx] += _c[local_idx + 64];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _c[local_idx] += _c[local_idx + 32];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _c[local_idx] += _c[local_idx + 16];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _c[local_idx] += _c[local_idx + 8];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _c[local_idx] += _c[local_idx + 4];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _c[local_idx] += _c[local_idx + 2];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_c, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_c, wGroup);
            #endif

            //Set C_tmp
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t group_idx = wGroup.get_group_id(0);
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     C_tmp[group_idx] = _c[0];
                }
            });
        }));
    });

    //Set C
    size_t grid_size_reduce = (grid_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(event_reduce_to_C_tmp);
        cgh.parallel_for_work_group(sycl::range<1>{grid_size_reduce}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            double _c[GPU_BLOCK_SIZE];
            // Set shared local memory _c
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < grid_size) {
                    _c[local_idx] = C_tmp[global_idx];
                } else {
                    _c[local_idx] = 0.0;
                }
            });

            // Reduce _c (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _c[local_idx] += _c[local_idx + 512];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _c[local_idx] += _c[local_idx + 256];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _c[local_idx] += _c[local_idx + 128];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _c[local_idx] += _c[local_idx + 64];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _c[local_idx] += _c[local_idx + 32];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _c[local_idx] += _c[local_idx + 16];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _c[local_idx] += _c[local_idx + 8];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _c[local_idx] += _c[local_idx + 4];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _c[local_idx] += _c[local_idx + 2];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_c, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_c, wGroup);
            #endif

            //Set C
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t group_idx = wGroup.get_group_id(0);
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     C[0] = _c[0];
                }
            });
        }));
    });
}

void gpu_matrix_multiply_LxM_by_MxN(sycl::queue q, size_t grid_size, double* __restrict__ C_tmp, double* __restrict__ C, double* __restrict__ A, double* __restrict__ B, size_t L, size_t M) {
    // C = B*A where [B] = LxM and [A] = MxN ???? FIXME this doesn't make sense
    gpu_matrix_multiply_MxN_by_Nx1(q, grid_size, C_tmp, C, B, A, M, L);
}

void gpu_normalize_population(sycl::queue q, size_t grid_size, double* __restrict__ population, double* __restrict__ normalization, double zeroth_moment, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size*genome_size) {
                    population[global_idx] *= zeroth_moment/normalization[global_idx/genome_size];
                }
            });
        }));
    });
}

void gpu_set_fitness(sycl::queue q, size_t grid_size, double* __restrict__ fitness_tmp, double* __restrict__ fitness, double* __restrict__ isf, double* __restrict__ isf_model, double* __restrict__ isf_error, size_t number_of_timeslices) {
    auto event_reduce_to_fitness_tmp = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            // Set shared local memory _f
            double _f[GPU_BLOCK_SIZE];
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < number_of_timeslices) {
                    _f[local_idx] = sycl::pown((isf[global_idx] - isf_model[global_idx])/isf_error[global_idx], 2);
                } else {
                    _f[local_idx] = 0.0;
                }
            });

            // Reduce _f (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _f[local_idx] += _f[local_idx + 512];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _f[local_idx] += _f[local_idx + 256];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _f[local_idx] += _f[local_idx + 128];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _f[local_idx] += _f[local_idx + 64];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _f[local_idx] += _f[local_idx + 32];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _f[local_idx] += _f[local_idx + 16];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _f[local_idx] += _f[local_idx + 8];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _f[local_idx] += _f[local_idx + 4];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _f[local_idx] += _f[local_idx + 2];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_f, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_f, wGroup);
            #endif

            //Set fitness_tmp
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t group_idx = wGroup.get_group_id(0);
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     fitness_tmp[group_idx] = _f[0];
                }
            });
        }));
    });

    size_t grid_size_reduce = (grid_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(event_reduce_to_fitness_tmp);
        cgh.parallel_for_work_group(sycl::range<1>{grid_size_reduce}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            double _f[GPU_BLOCK_SIZE];
            // Set shared local memory _f
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < grid_size) {
                    _f[local_idx] = fitness_tmp[global_idx];
                } else {
                    _f[local_idx] = 0.0;
                }
            });

            // Reduce _f (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _f[local_idx] += _f[local_idx + 512];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _f[local_idx] += _f[local_idx + 256];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _f[local_idx] += _f[local_idx + 128];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _f[local_idx] += _f[local_idx + 64];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _f[local_idx] += _f[local_idx + 32];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _f[local_idx] += _f[local_idx + 16];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _f[local_idx] += _f[local_idx + 8];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _f[local_idx] += _f[local_idx + 4];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _f[local_idx] += _f[local_idx + 2];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_f, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_f, wGroup);
            #endif

            //Set fitness
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t group_idx = wGroup.get_group_id(0);
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     fitness[0] = _f[0]/number_of_timeslices;
                }
            });
        }));
    });
}

void gpu_set_fitness_moments_reduced_chi_squared(sycl::queue q, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, double moment_error, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size) {
                    fitness[global_idx] += sycl::pown((moment - moments[global_idx])/moment_error, 2);
                }
            });
        }));
    });
}

void gpu_set_fitness_moments_chi_squared(sycl::queue q, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size) {
                    fitness[global_idx] += sycl::pown(moment - moments[global_idx], 2);
                }
            });
        }));
    });
}

void gpu_get_minimum_fitness(sycl::queue q, double* __restrict__ fitness, double* __restrict__ minimum_fitness, size_t population_size) {
    gpu_get_minimum(q, minimum_fitness, fitness, population_size);
}

void gpu_set_fitness_mean(sycl::queue q, size_t grid_size, double* __restrict__ fitness_mean_tmp, double* __restrict__ fitness_mean, double* __restrict__ fitness, size_t population_size) {
    auto event_reduce_to_fitness_mean_tmp = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            // Set shared local memory _fm
            double _fm[GPU_BLOCK_SIZE];
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < population_size) {
                    _fm[local_idx] = fitness[global_idx];
                } else {
                    _fm[local_idx] = 0.0;
                }
            });
            
            // Reduce _fm (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _fm[local_idx] += _fm[local_idx + 512];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _fm[local_idx] += _fm[local_idx + 256];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _fm[local_idx] += _fm[local_idx + 128];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _fm[local_idx] += _fm[local_idx + 64];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _fm[local_idx] += _fm[local_idx + 32];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _fm[local_idx] += _fm[local_idx + 16];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _fm[local_idx] += _fm[local_idx + 8];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _fm[local_idx] += _fm[local_idx + 4];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _fm[local_idx] += _fm[local_idx + 2];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_fm, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_fm, wGroup);
            #endif

            //Set fitness_mean_tmp
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t group_idx = wGroup.get_group_id(0);
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     fitness_mean_tmp[group_idx] = _fm[0];
                }
            });
        }));
    });

    size_t grid_size_reduce = (grid_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(event_reduce_to_fitness_mean_tmp);
        cgh.parallel_for_work_group(sycl::range<1>{grid_size_reduce}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            double _fm[GPU_BLOCK_SIZE];
            // Set shared local memory _fm
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < grid_size) {
                    _fm[local_idx] = fitness_mean_tmp[global_idx];
                } else {
                    _fm[local_idx] = 0.0;
                }
            });

            // Reduce _fm (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _fm[local_idx] += _fm[local_idx + 512];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _fm[local_idx] += _fm[local_idx + 256];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _fm[local_idx] += _fm[local_idx + 128];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _fm[local_idx] += _fm[local_idx + 64];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _fm[local_idx] += _fm[local_idx + 32];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _fm[local_idx] += _fm[local_idx + 16];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _fm[local_idx] += _fm[local_idx + 8];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _fm[local_idx] += _fm[local_idx + 4];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _fm[local_idx] += _fm[local_idx + 2];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_fm, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_fm, wGroup);
            #endif

            //Set fitness_mean
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t group_idx = wGroup.get_group_id(0);
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     fitness_mean[0] = _fm[0]/population_size;
                }
            });
        }));
    });
}

void gpu_set_fitness_squared_mean(sycl::queue q, size_t grid_size, double* __restrict__ fitness_squared_mean_tmp, double* __restrict__ fitness_squared_mean, double* __restrict__ fitness, size_t population_size) {
    auto event_reduce_to_fitness_squared_mean_tmp = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            // Set shared local memory _fsm
            double _fsm[GPU_BLOCK_SIZE];
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < population_size) {
                    _fsm[local_idx] = fitness[global_idx]*fitness[global_idx];
                } else {
                    _fsm[local_idx] = 0.0;
                }
            });
            
            // Reduce _fsm (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _fsm[local_idx] += _fsm[local_idx + 512];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _fsm[local_idx] += _fsm[local_idx + 256];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _fsm[local_idx] += _fsm[local_idx + 128];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _fsm[local_idx] += _fsm[local_idx + 64];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _fsm[local_idx] += _fsm[local_idx + 32];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _fsm[local_idx] += _fsm[local_idx + 16];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _fsm[local_idx] += _fsm[local_idx + 8];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _fsm[local_idx] += _fsm[local_idx + 4];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _fsm[local_idx] += _fsm[local_idx + 2];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_fsm, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_fsm, wGroup);
            #endif

            //Set fitness_squared_mean_tmp
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t group_idx = wGroup.get_group_id(0);
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     fitness_squared_mean_tmp[group_idx] = _fsm[0];
                }
            });
        }));
    });

    size_t grid_size_reduce = (grid_size + GPU_BLOCK_SIZE - 1) / GPU_BLOCK_SIZE;
    q.submit([&](sycl::handler& cgh) {
        cgh.depends_on(event_reduce_to_fitness_squared_mean_tmp);
        cgh.parallel_for_work_group(sycl::range<1>{grid_size_reduce}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            double _fsm[GPU_BLOCK_SIZE];
            // Set shared local memory _fsm
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                size_t local_idx = index.get_local_id(0);
                if (global_idx < grid_size) {
                    _fsm[local_idx] = fitness_squared_mean_tmp[global_idx];
                } else {
                    _fsm[local_idx] = 0.0;
                }
            });

            // Reduce _fsm (using shared local memory)
            #ifdef SYCL_MATMUL_WA
                // copy/paste code directly from gpu_reduce_add
                #if (GPU_BLOCK_SIZE >= 1024) && (SUB_GROUP_SIZE < 512)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 512) {
                            _fsm[local_idx] += _fsm[local_idx + 512];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 512) && (SUB_GROUP_SIZE < 256)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 256) {
                            _fsm[local_idx] += _fsm[local_idx + 256];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 256) && (SUB_GROUP_SIZE < 128)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 128) {
                            _fsm[local_idx] += _fsm[local_idx + 128];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 128) && (SUB_GROUP_SIZE < 64)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 64) {
                            _fsm[local_idx] += _fsm[local_idx + 64];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 64) && (SUB_GROUP_SIZE < 32)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 32) {
                            _fsm[local_idx] += _fsm[local_idx + 32];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 32) && (SUB_GROUP_SIZE < 16)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 16) {
                            _fsm[local_idx] += _fsm[local_idx + 16];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 16) && (SUB_GROUP_SIZE < 8)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 8) {
                            _fsm[local_idx] += _fsm[local_idx + 8];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 8) && (SUB_GROUP_SIZE < 4)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 4) {
                            _fsm[local_idx] += _fsm[local_idx + 4];
                        }
                    });
                #endif

                #if (GPU_BLOCK_SIZE >= 4) && (SUB_GROUP_SIZE < 2)
                    wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                        size_t local_idx = index.get_local_id(0);
                        if (local_idx < 2) {
                            _fsm[local_idx] += _fsm[local_idx + 2];
                        }
                    });
                #endif

                //Sub-group reduce
                wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                    size_t local_idx = index.get_local_id(0);
                    if (local_idx < SUB_GROUP_SIZE) {
                        sub_group_reduce_add(_fsm, local_idx);
                    }
                });
            #else
                gpu_reduce_add(_fsm, wGroup);
            #endif

            //Set fitness_squared_mean
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t local_idx = index.get_local_id(0);
                if (local_idx == 0) {
                     fitness_squared_mean[0] = _fsm[0]/population_size;
                }
            });
        }));
    });
}

void gpu_set_population_new(sycl::queue q, size_t grid_size, double* __restrict__ population_new, double* __restrict__ population_old, size_t* __restrict__ mutant_indices, double* __restrict__ differential_weights_new, bool* __restrict__ mutate_indices, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size*genome_size) {
                    size_t _i = global_idx/genome_size;
                    size_t _j = global_idx - _i*genome_size;
                    double F = differential_weights_new[_i];
                    size_t mutant_index1 = mutant_indices[3*_i];
                    size_t mutant_index2 = mutant_indices[3*_i + 1];
                    size_t mutant_index3 = mutant_indices[3*_i + 2];
                    bool mutate = mutate_indices[global_idx];
                    if (mutate) {
                        #ifdef ALLOW_NEGATIVE_SPECTRAL_WEIGHT
                            population_new[global_idx] = population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]);
                        #else
                            population_new[global_idx] = sycl::fabs( population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]) );
                        #endif
                    } else {
                        population_new[global_idx] = population_old[global_idx];
                    }
                }
            });
        }));
    });
}

void gpu_match_population_zero(sycl::queue q, size_t grid_size, double* __restrict__ population_negative_frequency, double* __restrict__ population_positive_frequency, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size) {
                    population_negative_frequency[global_idx*genome_size] = population_positive_frequency[global_idx*genome_size];
                }
            });
        }));
    });
}

void gpu_set_rejection_indices(sycl::queue q, size_t grid_size, bool* __restrict__ rejection_indices, double* __restrict__ fitness_new, double* __restrict__ fitness_old, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size) {
                    bool accept = fitness_new[global_idx] <= fitness_old[global_idx];
                    rejection_indices[global_idx] = accept;
                    if (accept) {
                        fitness_old[global_idx] = fitness_new[global_idx];
                    }
                }
            });
        }));
    });
}

void gpu_swap_control_parameters(sycl::queue q, size_t grid_size, double* __restrict__ control_parameter_old, double* __restrict__ control_parameter_new, bool* __restrict__ rejection_indices, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size) {
                    if (rejection_indices[global_idx]) {
                        control_parameter_old[global_idx] = control_parameter_new[global_idx];
                    }
                }
            });
        }));
    });
}

void gpu_swap_populations(sycl::queue q, size_t grid_size, double* __restrict__ population_old, double* __restrict__ population_new, bool* __restrict__ rejection_indices, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size*genome_size) {
                    size_t _i = global_idx/genome_size;
                    if (rejection_indices[_i]) {
                        population_old[global_idx] = population_new[global_idx];
                    }
                }
            });
        }));
    });
}

void gpu_set_crossover_probabilities_new(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ crossover_probabilities_new, double* __restrict__ crossover_probabilities_old, double self_adapting_crossover_probability, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size) {
                    if ((gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < self_adapting_crossover_probability) {
                        crossover_probabilities_new[global_idx] = (gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53;
                    } else {
                        crossover_probabilities_new[global_idx] = crossover_probabilities_old[global_idx];
                    }
                }
            });
        }));
    });
}

void gpu_set_differential_weights_new(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, double* __restrict__ differential_weights_new, double* __restrict__ differential_weights_old, double self_adapting_differential_weight_probability, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size) {
                    if ((gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < self_adapting_differential_weight_probability) {
                        differential_weights_new[global_idx] = 2.0*((gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53);
                    } else {
                        differential_weights_new[global_idx] = differential_weights_old[global_idx];
                    }
                }
            });
        }));
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
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size) {
                    gpu_set_mutant_indices(rng_state + 4*global_idx, mutant_indices + 3*global_idx, global_idx, population_size);
                }
            });
        }));
    });
}

void gpu_set_mutate_indices(sycl::queue q, size_t grid_size, uint64_t* __restrict__ rng_state, bool* __restrict__ mutate_indices, double* __restrict__ crossover_probabilities, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{grid_size}, sycl::range<1>{GPU_BLOCK_SIZE}, ([=](sycl::group<1> wGroup) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            wGroup.parallel_for_work_item([&](sycl::h_item<1> index) {
                size_t global_idx = index.get_global_id(0);
                if (global_idx < population_size*genome_size) {
                    size_t _i = global_idx/genome_size;
                    mutate_indices[global_idx] = (gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < crossover_probabilities[_i];
                }
            });
        }));
    });
}

#endif
