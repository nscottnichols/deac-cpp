#ifdef DEAC_TEST_WITH_NDEBUG
    #ifdef NDEBUG
        #undef NDEBUG
    #endif
    #define NDEBUG 1
#else
    #ifdef NDEBUG
        #undef NDEBUG
    #endif
#endif

#include "gpu_status.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

#ifdef DEAC_TEST_WITH_NDEBUG
    #ifndef NDEBUG
        #error "NDEBUG status-check regression did not define NDEBUG"
    #endif
#else
    #ifdef NDEBUG
        #error "Debug status-check regression did not clear NDEBUG"
    #endif
#endif

namespace {
enum class fake_status {
    success = 0,
    runtime_failure = 17,
    blas_failure = 29,
};

fake_status counted_status(int& evaluations, fake_status result) {
    ++evaluations;
    return result;
}

struct counted_description {
    int* evaluations;
    const char* text;

    const char* operator()(fake_status) const {
        ++*evaluations;
        return text;
    }
};

bool contains(const std::string& text, const std::string& expected) {
    if (text.find(expected) == std::string::npos) {
        std::cerr << "missing diagnostic fragment " << expected
                  << " in: " << text << '\n';
        return false;
    }
    return true;
}

template<typename Describe>
bool check_success(
        const char* category,
        Describe describe,
        int& evaluations,
        int& description_evaluations) {
    DEAC_GPU_STATUS_CHECK(
            counted_status(evaluations, fake_status::success),
            fake_status::success, "FAKE", category, describe);
    if (evaluations != 1 || description_evaluations != 0) {
        std::cerr << category << " success evaluated operation "
                  << evaluations << " times and description "
                  << description_evaluations << " times\n";
        return false;
    }
    return true;
}

template<typename Describe>
bool check_failure(
        const char* category,
        fake_status failure,
        int expected_code,
        Describe describe,
        int& description_evaluations,
        int expected_description_evaluations,
        const char* expected_description,
        std::string& diagnostic) {
    int evaluations = 0;
    try {
        DEAC_GPU_STATUS_CHECK(
                counted_status(evaluations, failure),
                fake_status::success, "FAKE", category, describe);
        std::cerr << category << " failure did not throw\n";
        return false;
    } catch (const deac_gpu_status::status_error& error) {
        diagnostic = error.what();
    }

    bool valid = true;
    if (evaluations != 1
            || description_evaluations != expected_description_evaluations) {
        std::cerr << category << " failure evaluated operation "
                  << evaluations << " times and description "
                  << description_evaluations << " times\n";
        valid = false;
    }
    valid = contains(diagnostic, std::string("FAKE ") + category
            + " call failed: counted_status(evaluations, failure)") && valid;
    std::string status_fragment =
            "status=" + std::to_string(expected_code);
    if (expected_description == nullptr) {
        status_fragment += "; location=";
    } else {
        status_fragment +=
                std::string(" (") + expected_description + "); location=";
    }
    valid = contains(diagnostic, status_fragment) && valid;
    valid = contains(diagnostic, "gpu_status_test.cpp:") && valid;
    return valid;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 1) {
        int runtime_evaluations = 0;
        int blas_evaluations = 0;
        int runtime_descriptions = 0;
        int blas_descriptions = 0;
        const bool runtime_valid = check_success(
                "runtime",
                counted_description{
                        &runtime_descriptions,
                        "unused runtime description"},
                runtime_evaluations, runtime_descriptions);
        const bool blas_valid = check_success(
                "BLAS", deac_gpu_status::no_description{},
                blas_evaluations, blas_descriptions);
        if (!runtime_valid || !blas_valid) {
            return EXIT_FAILURE;
        }
        std::cout << "success checks passed; runtime_evaluations=1; "
                     "blas_evaluations=1\n";
        return EXIT_SUCCESS;
    }

    if (argc != 2 || std::string(argv[1]) != "--failure") {
        std::cerr << "usage: gpu_status_test [--failure]\n";
        return 2;
    }

    std::string runtime_diagnostic;
    std::string blas_diagnostic;
    int runtime_descriptions = 0;
    int blas_descriptions = 0;
    const bool runtime_valid = check_failure(
            "runtime", fake_status::runtime_failure, 17,
            counted_description{
                    &runtime_descriptions, "controlled runtime failure"},
            runtime_descriptions, 1, "controlled runtime failure",
            runtime_diagnostic);
    const bool blas_valid = check_failure(
            "BLAS", fake_status::blas_failure, 29,
            deac_gpu_status::no_description{},
            blas_descriptions, 0, nullptr, blas_diagnostic);
    if (!runtime_valid || !blas_valid) {
        return 2;
    }

    std::cerr << runtime_diagnostic << '\n' << blas_diagnostic << '\n'
              << "controlled failure checks passed; runtime_evaluations=1; "
                 "blas_evaluations=1\n";
    return EXIT_FAILURE;
}
