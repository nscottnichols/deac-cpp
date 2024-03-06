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
                 C[0] = _c[0];
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

void gpu_normalize_population(sycl::queue q, size_t grid_size, double* __restrict__ population, double* __restrict__ normalization, double zeroth_moment, size_t population_size, size_t genome_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size*genome_size) {
                population[global_idx] *= zeroth_moment/normalization[global_idx/genome_size];
            }
        });
    });
}

void gpu_set_fitness(sycl::queue q, double* __restrict__ fitness, double* __restrict__ isf, double* __restrict__ isf_model, double* __restrict__ isf_error, size_t number_of_timeslices) {
    q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<double, 1> _f(sycl::range<1>(GPU_BLOCK_SIZE), cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            // Set shared local memory _f
            size_t local_idx = item.get_local_id(0);
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
            sycl::group_barrier(item.get_group());

            // Reduce _f (using shared local memory)
            gpu_reduce_add(_f.get_pointer(), item);

            //Set fitness
            if (local_idx == 0) {
                 fitness[0] += _f[0]/number_of_timeslices;
            }
        });
    });
}

void gpu_set_fitness_moments_reduced_chi_squared(sycl::queue q, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, double moment_error, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size) {
                fitness[global_idx] += sycl::pown((moment - moments[global_idx])/moment_error, 2);
            }
        });
    });
}

void gpu_set_fitness_moments_chi_squared(sycl::queue q, size_t grid_size, double* __restrict__ fitness, double* __restrict__ moments, double moment, size_t population_size) {
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grid_size*GPU_BLOCK_SIZE), sycl::range<1>(GPU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] {
            size_t global_idx = item.get_global_id(0);
            if (global_idx < population_size) {
                fitness[global_idx] += sycl::pown(moment - moments[global_idx], 2);
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
                        population_new[global_idx] = sycl::fabs(population_old[mutant_index1*genome_size + _j] + F*(population_old[mutant_index2*genome_size + _j] - population_old[mutant_index3*genome_size + _j]));
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
                population_negative_frequency[global_idx*genome_size] = population_positive_frequency[global_idx*genome_size];
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
                size_t _i = global_idx/genome_size;
                mutate_indices[global_idx] = (gpu_xoshiro256p_next(rng_state + 4*global_idx) >> 11) * 0x1.0p-53 < crossover_probabilities[_i];
            }
        });
    });
}
#endif
