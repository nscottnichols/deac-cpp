cmake_minimum_required(VERSION 3.18)
project(deac_hipblas_link_contract_fixture LANGUAGES CXX)

foreach(_deac_required_variable IN ITEMS
        DEAC_HIPBLAS_MODULE
        DEAC_HIPBLAS_TEST_MODE)
    if(NOT DEFINED ${_deac_required_variable}
            OR "${${_deac_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_deac_required_variable} is required.")
    endif()
endforeach()

set(_deac_hipblas_consumers
    deac.e
    deac_gpu_normalization_test
    deac_invalid_normalization_evolution_test_exe)
foreach(_deac_consumer IN LISTS _deac_hipblas_consumers)
    add_executable("${_deac_consumer}" EXCLUDE_FROM_ALL dummy.cpp)
endforeach()

if(DEAC_HIPBLAS_TEST_MODE STREQUAL "imported_target")
    set(GPU_BACKEND hip)
    set(USE_BLAS ON)
    set(hipblas_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/mock-package"
        CACHE PATH "" FORCE)
elseif(DEAC_HIPBLAS_TEST_MODE STREQUAL "compatibility_library")
    set(GPU_BACKEND hip)
    set(USE_BLAS ON)
    set(CMAKE_DISABLE_FIND_PACKAGE_hipblas TRUE)
    set(DEAC_HIPBLAS_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/mock-include"
        CACHE PATH "" FORCE)
    set(_deac_mock_library_dir
        "${CMAKE_CURRENT_BINARY_DIR}/mock-library")
    file(MAKE_DIRECTORY "${_deac_mock_library_dir}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/mock-library/libhipblas.a"
        "${_deac_mock_library_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}hipblas${CMAKE_STATIC_LIBRARY_SUFFIX}"
        COPYONLY)
    list(PREPEND CMAKE_LIBRARY_PATH
        "${_deac_mock_library_dir}")
elseif(DEAC_HIPBLAS_TEST_MODE STREQUAL "missing_package")
    set(GPU_BACKEND hip)
    set(USE_BLAS ON)
    set(CMAKE_DISABLE_FIND_PACKAGE_hipblas TRUE)
    set(DEAC_HIPBLAS_INCLUDE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/missing/include"
        CACHE PATH "" FORCE)
    set(DEAC_HIPBLAS_LIBRARY
        "${CMAKE_CURRENT_BINARY_DIR}/missing/lib/libhipblas.so"
        CACHE FILEPATH "" FORCE)
elseif(DEAC_HIPBLAS_TEST_MODE STREQUAL "header_only")
    set(GPU_BACKEND hip)
    set(USE_BLAS ON)
    set(CMAKE_DISABLE_FIND_PACKAGE_hipblas TRUE)
    set(DEAC_HIPBLAS_INCLUDE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/mock-include"
        CACHE PATH "" FORCE)
    set(DEAC_HIPBLAS_LIBRARY
        "${CMAKE_CURRENT_BINARY_DIR}/missing/lib/libhipblas.so"
        CACHE FILEPATH "" FORCE)
elseif(DEAC_HIPBLAS_TEST_MODE STREQUAL "blas_off")
    set(GPU_BACKEND hip)
    set(USE_BLAS OFF)
    set(CMAKE_DISABLE_FIND_PACKAGE_hipblas TRUE)
elseif(DEAC_HIPBLAS_TEST_MODE STREQUAL "non_hip")
    set(GPU_BACKEND sycl)
    set(USE_BLAS ON)
    set(CMAKE_DISABLE_FIND_PACKAGE_hipblas TRUE)
else()
    message(FATAL_ERROR
        "Unknown DEAC_HIPBLAS_TEST_MODE '${DEAC_HIPBLAS_TEST_MODE}'.")
endif()

include("${DEAC_HIPBLAS_MODULE}")

# Apply the function twice deliberately.  The contract must remain idempotent:
# each real consumer receives exactly one direct dependency.
deac_target_link_hipblas(${_deac_hipblas_consumers})
deac_target_link_hipblas(${_deac_hipblas_consumers})

function(_deac_assert_link_count target_name property_name link_name expected_count)
    get_target_property(_deac_links "${target_name}" "${property_name}")
    set(_deac_count 0)
    foreach(_deac_link IN LISTS _deac_links)
        if("${_deac_link}" STREQUAL "${link_name}")
            math(EXPR _deac_count "${_deac_count} + 1")
        endif()
    endforeach()
    if(NOT _deac_count EQUAL expected_count)
        message(FATAL_ERROR
            "Expected ${target_name} ${property_name} to contain '${link_name}' "
            "${expected_count} time(s), got ${_deac_count}: ${_deac_links}")
    endif()
endfunction()

if(DEAC_HIPBLAS_TEST_MODE STREQUAL "imported_target"
        OR DEAC_HIPBLAS_TEST_MODE STREQUAL "compatibility_library")
    if(NOT TARGET deac::hipblas)
        message(FATAL_ERROR "The project-owned deac::hipblas contract is missing.")
    endif()
    if(DEAC_HIPBLAS_TEST_MODE STREQUAL "imported_target")
        set(_deac_expected_provider roc::hipblas)
    else()
        set(_deac_expected_provider deac_hipblas_compat)
        if(NOT TARGET deac_hipblas_compat)
            message(FATAL_ERROR
                "The validated compatibility hipBLAS target is missing.")
        endif()
        get_target_property(_deac_compat_location
            deac_hipblas_compat IMPORTED_LOCATION)
        get_target_property(_deac_compat_includes
            deac_hipblas_compat INTERFACE_INCLUDE_DIRECTORIES)
        if(NOT EXISTS "${_deac_compat_location}"
                OR IS_DIRECTORY "${_deac_compat_location}")
            message(FATAL_ERROR
                "The compatibility target has no actual library: "
                "${_deac_compat_location}")
        endif()
        if(NOT _deac_compat_includes STREQUAL
                "${CMAKE_CURRENT_SOURCE_DIR}/mock-include")
            message(FATAL_ERROR
                "The compatibility target has unexpected includes: "
                "${_deac_compat_includes}")
        endif()
    endif()
    _deac_assert_link_count(
        deac_hipblas_link_contract INTERFACE_LINK_LIBRARIES
        "${_deac_expected_provider}" 1)
    foreach(_deac_consumer IN LISTS _deac_hipblas_consumers)
        _deac_assert_link_count(
            "${_deac_consumer}" LINK_LIBRARIES deac::hipblas 1)
        get_property(_deac_contract_applied
            TARGET "${_deac_consumer}"
            PROPERTY DEAC_HIPBLAS_LINK_CONTRACT_APPLIED)
        if(NOT _deac_contract_applied)
            message(FATAL_ERROR
                "${_deac_consumer} did not record the hipBLAS contract.")
        endif()
    endforeach()
elseif(DEAC_HIPBLAS_TEST_MODE STREQUAL "blas_off"
        OR DEAC_HIPBLAS_TEST_MODE STREQUAL "non_hip")
    if(TARGET deac_hipblas_link_contract OR TARGET deac::hipblas)
        message(FATAL_ERROR
            "${DEAC_HIPBLAS_TEST_MODE} unexpectedly created a hipBLAS contract.")
    endif()
    foreach(_deac_consumer IN LISTS _deac_hipblas_consumers)
        get_target_property(_deac_links "${_deac_consumer}" LINK_LIBRARIES)
        if(_deac_links)
            message(FATAL_ERROR
                "${DEAC_HIPBLAS_TEST_MODE} unexpectedly linked "
                "${_deac_consumer}: ${_deac_links}")
        endif()
        get_property(_deac_contract_was_set
            TARGET "${_deac_consumer}"
            PROPERTY DEAC_HIPBLAS_LINK_CONTRACT_APPLIED
            SET)
        if(_deac_contract_was_set)
            message(FATAL_ERROR
                "${DEAC_HIPBLAS_TEST_MODE} marked ${_deac_consumer} as linked.")
        endif()
    endforeach()
endif()
