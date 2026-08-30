#pragma once

#include <string_view>

namespace deac_build_identity {

std::string_view semantic_version() noexcept;
std::string_view canonical_json() noexcept;

}  // namespace deac_build_identity
