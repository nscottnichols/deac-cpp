foreach(_required_variable
        DEAC_BUILD_IDENTITY_SOURCE_ROOT
        DEAC_BUILD_IDENTITY_OUTPUT_HEADER
        DEAC_BUILD_IDENTITY_OUTPUT_RECEIPT)
    if(NOT DEFINED ${_required_variable} OR "${${_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_required_variable} is required")
    endif()
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/DeacBuildIdentity.cmake")
_deac_compute_build_identity("${DEAC_BUILD_IDENTITY_SOURCE_ROOT}")

get_filename_component(
    _header_directory "${DEAC_BUILD_IDENTITY_OUTPUT_HEADER}" DIRECTORY)
get_filename_component(
    _receipt_directory "${DEAC_BUILD_IDENTITY_OUTPUT_RECEIPT}" DIRECTORY)
file(MAKE_DIRECTORY "${_header_directory}" "${_receipt_directory}")
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/deac_build_identity_data.hpp.in"
    "${DEAC_BUILD_IDENTITY_OUTPUT_HEADER}"
    @ONLY)
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/deac_build_identity.json.in"
    "${DEAC_BUILD_IDENTITY_OUTPUT_RECEIPT}"
    @ONLY)
