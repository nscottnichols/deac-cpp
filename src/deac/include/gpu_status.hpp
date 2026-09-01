#pragma once

#include <sstream>
#include <stdexcept>

namespace deac_gpu_status {
class status_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct no_description {
    template<typename Status>
    const char* operator()(Status) const noexcept {
        return nullptr;
    }
};

template<typename Status, typename Success, typename Describe>
void check(
        Status status,
        Success success,
        const char* backend,
        const char* category,
        const char* expression,
        const char* file,
        int line,
        Describe describe) {
    if (status == success) {
        return;
    }

    std::ostringstream message;
    message << backend << ' ' << category << " call failed: "
            << expression << "; status=" << static_cast<long long>(status);
    const char* description = describe(status);
    if (description != nullptr && description[0] != '\0') {
        message << " (" << description << ')';
    }
    message << "; location=" << file << ':' << line;
    throw status_error(message.str());
}
} // namespace deac_gpu_status

// The operation expression occurs exactly once outside standard assert, so it
// is evaluated identically whether or not NDEBUG is defined.
#define DEAC_GPU_STATUS_CHECK(                                      \
        expression, success, backend, category, describe)          \
    ::deac_gpu_status::check(                                      \
            (expression), (success), (backend), (category),        \
            #expression, __FILE__, __LINE__, (describe))
