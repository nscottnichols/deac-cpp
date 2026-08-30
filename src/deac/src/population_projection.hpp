#pragma once

#include <cstddef>

namespace deac_numerics {

// Accumulate the projection of each population row onto every kernel row.
//
// projection is a population_size-by-output_rows row-major matrix, kernel is
// output_rows-by-genome_size, and population is
// population_size-by-genome_size. All three ranges must be valid for those
// positive dimensions and must not overlap. Their element-count products must
// fit in std::size_t.
inline void accumulate_population_projection(
        double* projection,
        const double* kernel,
        const double* population,
        std::size_t output_rows,
        std::size_t genome_size,
        std::size_t population_size) {
    for (std::size_t population_index=0;
            population_index<population_size;
            ++population_index) {
        double* const projection_row = projection + population_index*output_rows;
        const double* const population_row = population + population_index*genome_size;

        std::size_t output_row=0;
        for (; output_rows - output_row >= 4; output_row += 4) {
            const double* const kernel_row0 = kernel + output_row*genome_size;
            const double* const kernel_row1 = kernel_row0 + genome_size;
            const double* const kernel_row2 = kernel_row1 + genome_size;
            const double* const kernel_row3 = kernel_row2 + genome_size;
            double projection0 = projection_row[output_row];
            double projection1 = projection_row[output_row + 1];
            double projection2 = projection_row[output_row + 2];
            double projection3 = projection_row[output_row + 3];

            for (std::size_t genome_index=0;
                    genome_index<genome_size;
                    ++genome_index) {
                const double population_value = population_row[genome_index];
                projection0 += kernel_row0[genome_index]*population_value;
                projection1 += kernel_row1[genome_index]*population_value;
                projection2 += kernel_row2[genome_index]*population_value;
                projection3 += kernel_row3[genome_index]*population_value;
            }

            projection_row[output_row] = projection0;
            projection_row[output_row + 1] = projection1;
            projection_row[output_row + 2] = projection2;
            projection_row[output_row + 3] = projection3;
        }

        for (; output_row<output_rows; ++output_row) {
            const double* const kernel_row = kernel + output_row*genome_size;
            double accumulated_projection = projection_row[output_row];
            for (std::size_t genome_index=0;
                    genome_index<genome_size;
                    ++genome_index) {
                accumulated_projection +=
                        kernel_row[genome_index]*population_row[genome_index];
            }
            projection_row[output_row] = accumulated_projection;
        }
    }
}

} // namespace deac_numerics
