#include "population_projection.hpp"

#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void reference_accumulate_population_projection(
        double* projection,
        const double* kernel,
        const double* population,
        std::size_t output_rows,
        std::size_t genome_size,
        std::size_t population_size) {
    for (std::size_t population_index=0;
            population_index<population_size;
            ++population_index) {
        for (std::size_t output_row=0; output_row<output_rows; ++output_row) {
            for (std::size_t genome_index=0;
                    genome_index<genome_size;
                    ++genome_index) {
                projection[population_index*output_rows + output_row] +=
                        kernel[output_row*genome_size + genome_index]
                        *population[population_index*genome_size + genome_index];
            }
        }
    }
}

double initial_projection_value(std::size_t index) {
    const double magnitude = static_cast<double>((index*7)%19 + 1)/32.0;
    return index%2 == 0 ? magnitude : -magnitude;
}

double kernel_value(std::size_t index) {
    const double magnitude = static_cast<double>((index*5)%23 + 1)/16.0;
    return index%3 == 0 ? -magnitude : magnitude;
}

double population_value(std::size_t index) {
    const double magnitude = static_cast<double>((index*11)%29 + 1)/8.0;
    return index%5 == 0 ? -magnitude : magnitude;
}

bool check_dimensions(
        std::size_t output_rows,
        std::size_t genome_size,
        std::size_t population_size) {
    std::vector<double> kernel(output_rows*genome_size);
    std::vector<double> population(population_size*genome_size);
    std::vector<double> expected(population_size*output_rows);

    for (std::size_t index=0; index<kernel.size(); ++index) {
        kernel[index] = kernel_value(index);
    }
    for (std::size_t index=0; index<population.size(); ++index) {
        population[index] = population_value(index);
    }
    for (std::size_t index=0; index<expected.size(); ++index) {
        expected[index] = initial_projection_value(index);
    }

    std::vector<double> actual = expected;
    reference_accumulate_population_projection(
            expected.data(), kernel.data(), population.data(),
            output_rows, genome_size, population_size);
    deac_numerics::accumulate_population_projection(
            actual.data(), kernel.data(), population.data(),
            output_rows, genome_size, population_size);

    for (std::size_t index=0; index<actual.size(); ++index) {
        if (actual[index] != expected[index]) {
            std::cerr << "projection mismatch for L=" << output_rows
                      << ", M=" << genome_size
                      << ", N=" << population_size
                      << " at flat index " << index
                      << ": expected " << std::hexfloat << expected[index]
                      << ", got " << actual[index] << '\n';
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    constexpr std::array<std::size_t, 9> output_row_counts{1, 2, 3, 4, 5, 6, 7, 8, 9};
    constexpr std::array<std::size_t, 4> genome_sizes{1, 2, 3, 7};
    constexpr std::array<std::size_t, 3> population_sizes{1, 3, 6};

    for (const std::size_t output_rows : output_row_counts) {
        for (const std::size_t genome_size : genome_sizes) {
            for (const std::size_t population_size : population_sizes) {
                if (!check_dimensions(output_rows, genome_size, population_size)) {
                    return 1;
                }
            }
        }
    }

    return 0;
}
