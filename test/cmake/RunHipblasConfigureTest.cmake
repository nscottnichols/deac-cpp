cmake_minimum_required(VERSION 3.18)

foreach(_deac_required_variable IN ITEMS
        DEAC_TEST_CMAKE_COMMAND
        DEAC_TEST_SOURCE_DIR
        DEAC_TEST_BINARY_DIR
        DEAC_TEST_CXX_COMPILER
        DEAC_TEST_GENERATOR
        DEAC_TEST_HIPBLAS_MODULE
        DEAC_TEST_MODE
        DEAC_TEST_EXPECT_SUCCESS)
    if(NOT DEFINED ${_deac_required_variable}
            OR "${${_deac_required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "${_deac_required_variable} is required for the hipBLAS configure test.")
    endif()
endforeach()

file(REMOVE_RECURSE "${DEAC_TEST_BINARY_DIR}")
set(_deac_configure_command
        "${DEAC_TEST_CMAKE_COMMAND}"
        -G "${DEAC_TEST_GENERATOR}"
        -S "${DEAC_TEST_SOURCE_DIR}"
        -B "${DEAC_TEST_BINARY_DIR}"
        "-DCMAKE_CXX_COMPILER:FILEPATH=${DEAC_TEST_CXX_COMPILER}"
        "-DDEAC_HIPBLAS_MODULE:FILEPATH=${DEAC_TEST_HIPBLAS_MODULE}"
        "-DDEAC_HIPBLAS_TEST_MODE:STRING=${DEAC_TEST_MODE}")
if(DEFINED DEAC_TEST_MAKE_PROGRAM
        AND NOT "${DEAC_TEST_MAKE_PROGRAM}" STREQUAL "")
    list(APPEND _deac_configure_command
        "-DCMAKE_MAKE_PROGRAM:FILEPATH=${DEAC_TEST_MAKE_PROGRAM}")
endif()
if(DEFINED DEAC_TEST_GENERATOR_PLATFORM
        AND NOT "${DEAC_TEST_GENERATOR_PLATFORM}" STREQUAL "")
    list(APPEND _deac_configure_command
        -A "${DEAC_TEST_GENERATOR_PLATFORM}")
endif()
if(DEFINED DEAC_TEST_GENERATOR_TOOLSET
        AND NOT "${DEAC_TEST_GENERATOR_TOOLSET}" STREQUAL "")
    list(APPEND _deac_configure_command
        -T "${DEAC_TEST_GENERATOR_TOOLSET}")
endif()
if(DEFINED DEAC_TEST_GENERATOR_INSTANCE
        AND NOT "${DEAC_TEST_GENERATOR_INSTANCE}" STREQUAL "")
    list(APPEND _deac_configure_command
        "-DCMAKE_GENERATOR_INSTANCE:STRING=${DEAC_TEST_GENERATOR_INSTANCE}")
endif()
execute_process(
    COMMAND ${_deac_configure_command}
    RESULT_VARIABLE _deac_configure_result
    OUTPUT_VARIABLE _deac_configure_stdout
    ERROR_VARIABLE _deac_configure_stderr)
set(_deac_configure_output
    "${_deac_configure_stdout}\n${_deac_configure_stderr}")

if(DEAC_TEST_EXPECT_SUCCESS)
    if(NOT _deac_configure_result EQUAL 0)
        message(FATAL_ERROR
            "hipBLAS ${DEAC_TEST_MODE} fixture unexpectedly failed with "
            "${_deac_configure_result}:\n${_deac_configure_output}")
    endif()
else()
    if(_deac_configure_result EQUAL 0)
        message(FATAL_ERROR
            "hipBLAS ${DEAC_TEST_MODE} fixture unexpectedly configured successfully.")
    endif()
    set(_deac_expected_failure_text
        "USE_BLAS=ON"
        "GPU_BACKEND=hip"
        "roc::hipblas"
        "CMAKE_PREFIX_PATH"
        "DEAC_HIPBLAS_LIBRARY")
    if(DEAC_TEST_MODE STREQUAL "header_only")
        list(APPEND _deac_expected_failure_text
            "an actual hipBLAS library")
    elseif(DEAC_TEST_MODE STREQUAL "missing_package")
        list(APPEND _deac_expected_failure_text
            "hipblas/hipblas.h")
    endif()
    foreach(_deac_expected_text IN LISTS _deac_expected_failure_text)
        string(FIND "${_deac_configure_output}"
            "${_deac_expected_text}" _deac_expected_text_position)
        if(_deac_expected_text_position EQUAL -1)
            message(FATAL_ERROR
                "hipBLAS ${DEAC_TEST_MODE} failure did not contain actionable text "
                "'${_deac_expected_text}':\n${_deac_configure_output}")
        endif()
    endforeach()
endif()
