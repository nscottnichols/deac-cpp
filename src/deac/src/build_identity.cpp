#include "build_identity.hpp"

#include "deac_build_identity_data.hpp"

namespace deac_build_identity {

std::string_view semantic_version() noexcept {
    return generated::semantic_version;
}

std::string_view canonical_json() noexcept {
    return generated::canonical_json;
}

}  // namespace deac_build_identity
