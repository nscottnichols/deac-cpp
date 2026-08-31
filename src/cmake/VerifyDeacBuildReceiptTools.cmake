cmake_minimum_required(VERSION 3.27)

foreach(_required_variable
        DEAC_BUILD_RECEIPT_LANGUAGES
        DEAC_BUILD_RECEIPT_CMAKE_PATH
        DEAC_BUILD_RECEIPT_CMAKE_REAL_PATH
        DEAC_BUILD_RECEIPT_CMAKE_SHA256)
    if(NOT DEFINED ${_required_variable} OR
            "${${_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_required_variable} is required")
    endif()
endforeach()

function(_deac_verify_build_receipt_tool label path expected_real expected_sha256)
    get_filename_component(_real_path "${path}" REALPATH)
    if(NOT EXISTS "${_real_path}" OR IS_DIRECTORY "${_real_path}")
        message(FATAL_ERROR
            "build-receipt ${label} is no longer a regular file: ${path}")
    endif()
    file(SHA256 "${_real_path}" _sha256)
    if(NOT _real_path STREQUAL "${expected_real}" OR
            NOT _sha256 STREQUAL "${expected_sha256}")
        message(FATAL_ERROR
            "build-receipt ${label} changed after configuration; "
            "rerun CMake before building")
    endif()
endfunction()

_deac_verify_build_receipt_tool(
    "CMake executable"
    "${DEAC_BUILD_RECEIPT_CMAKE_PATH}"
    "${DEAC_BUILD_RECEIPT_CMAKE_REAL_PATH}"
    "${DEAC_BUILD_RECEIPT_CMAKE_SHA256}")

string(REPLACE "," ";" _languages "${DEAC_BUILD_RECEIPT_LANGUAGES}")
foreach(_language IN LISTS _languages)
    set(_path_variable
        "DEAC_BUILD_RECEIPT_CONFIGURED_${_language}_PATH")
    set(_real_path_variable
        "DEAC_BUILD_RECEIPT_CONFIGURED_${_language}_REAL_PATH")
    set(_sha256_variable
        "DEAC_BUILD_RECEIPT_CONFIGURED_${_language}_SHA256")
    foreach(_variable
            ${_path_variable} ${_real_path_variable} ${_sha256_variable})
        if(NOT DEFINED ${_variable} OR "${${_variable}}" STREQUAL "")
            message(FATAL_ERROR "${_variable} is required")
        endif()
    endforeach()
    _deac_verify_build_receipt_tool(
        "${_language} compiler"
        "${${_path_variable}}"
        "${${_real_path_variable}}"
        "${${_sha256_variable}}")
endforeach()
