#include "moment_fitness.hpp"

#include <cmath>
#include <iostream>

int main() {
    if (deac_numerics::scalar_chi_square_penalty(2.0, 0.0, 1.0) != 4.0) {
        std::cerr << "zero-target first-moment penalty is not finite and squared\n";
        return 1;
    }
    if (deac_numerics::scalar_chi_square_penalty(2.0, 1.0, 0.5) != 4.0) {
        std::cerr << "scalar penalty does not divide by the error before squaring\n";
        return 1;
    }
    if (!std::isfinite(
                deac_numerics::scalar_chi_square_penalty(0.0, 0.0, 1.0))) {
        std::cerr << "matching zero moment produced a non-finite penalty\n";
        return 1;
    }
    return 0;
}
