cmake_minimum_required(VERSION 3.18)

foreach(_deac_required_variable IN ITEMS
        DEAC_TEST_CMAKE_COMMAND
        DEAC_TEST_PROJECT_SOURCE_DIR
        DEAC_TEST_BINARY_DIR
        DEAC_TEST_CXX_COMPILER
        DEAC_TEST_GENERATOR
        DEAC_TEST_MOCK_INCLUDE_DIR
        DEAC_TEST_MOCK_PACKAGE_DIR)
    if(NOT DEFINED ${_deac_required_variable}
            OR "${${_deac_required_variable}}" STREQUAL "")
        message(FATAL_ERROR
            "${_deac_required_variable} is required for the hipBLAS project configure test.")
    endif()
endforeach()

file(REMOVE_RECURSE "${DEAC_TEST_BINARY_DIR}")
set(_deac_graph "${DEAC_TEST_BINARY_DIR}/targets.dot")
set(_deac_configure_command
        "${DEAC_TEST_CMAKE_COMMAND}"
        "--graphviz=${_deac_graph}"
        -G "${DEAC_TEST_GENERATOR}"
        -S "${DEAC_TEST_PROJECT_SOURCE_DIR}"
        -B "${DEAC_TEST_BINARY_DIR}"
        "-DCMAKE_CXX_COMPILER:FILEPATH=${DEAC_TEST_CXX_COMPILER}"
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DGPU_BACKEND:STRING=hip
        -DUSE_BLAS:BOOL=ON
        "-DHIP_RUNTIME_INCLUDE_DIR:PATH=${DEAC_TEST_MOCK_INCLUDE_DIR}"
        "-Dhipblas_DIR:PATH=${DEAC_TEST_MOCK_PACKAGE_DIR}")
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
if(NOT _deac_configure_result EQUAL 0)
    message(FATAL_ERROR
        "The mock HIP+BLAS project configure failed with ${_deac_configure_result}:\n"
        "${_deac_configure_stdout}\n${_deac_configure_stderr}")
endif()
if(NOT EXISTS "${_deac_graph}")
    message(FATAL_ERROR
        "The mock HIP+BLAS project configure did not produce ${_deac_graph}.")
endif()
file(READ "${_deac_graph}" _deac_graph_text)

function(_deac_assert_literal_count text_variable needle expected_count)
    set(_deac_remaining "${${text_variable}}")
    set(_deac_count 0)
    string(LENGTH "${needle}" _deac_needle_length)
    while(TRUE)
        string(FIND "${_deac_remaining}" "${needle}" _deac_position)
        if(_deac_position EQUAL -1)
            break()
        endif()
        math(EXPR _deac_next "${_deac_position} + ${_deac_needle_length}")
        string(SUBSTRING "${_deac_remaining}" ${_deac_next} -1 _deac_remaining)
        math(EXPR _deac_count "${_deac_count} + 1")
    endwhile()
    if(NOT _deac_count EQUAL expected_count)
        message(FATAL_ERROR
            "Expected '${needle}' ${expected_count} time(s) in the target graph, "
            "got ${_deac_count}.\n${_deac_graph_text}")
    endif()
endfunction()

foreach(_deac_consumer IN ITEMS
        deac.e
        deac_gpu_normalization_test
        deac_gpu_fitness_test
        deac_invalid_normalization_evolution_test_exe)
    _deac_assert_literal_count(_deac_graph_text
        "${_deac_consumer} -> deac_hipblas_link_contract" 1)
    _deac_assert_literal_count(_deac_graph_text
        "${_deac_consumer} -> roc::hipblas" 0)
endforeach()
_deac_assert_literal_count(_deac_graph_text
    "deac_hipblas_link_contract -> roc::hipblas" 1)
