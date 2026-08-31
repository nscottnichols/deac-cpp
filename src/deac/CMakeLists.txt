cmake_minimum_required(VERSION 3.27.0)

project(deac.e LANGUAGES CXX)

include("${CMAKE_SOURCE_DIR}/cmake/DeacHipblas.cmake")

# Set c++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Check for static build
if(STATIC)
    message("Static build specified, setting library suffixes to ${CMAKE_STATIC_LIBRARY_SUFFIX}.")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(BUILD_SHARED_LIBS OFF)
    list(APPEND CMAKE_EXE_LINKER_FLAGS "-static")
    list(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG "-static")
endif()

# Optional build features
option(USE_SYCL_MATMUL_WA "Enable SYCL matmul workaround." OFF)
option(USE_ALLOW_NEGATIVE_SPECTRAL_WEIGHT "Allow negative spectral weights." OFF)
option(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF "Build the bosonic detailed-balance DSF kernel variant." OFF)
option(SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION "Enable single-particle fermionic spectral function behavior." OFF)
option(USE_BLAS "Use backend BLAS libraries for GPU matrix operations." OFF)

if(USE_SYCL_MATMUL_WA)
    add_definitions(-D SYCL_MATMUL_WA=1)
endif()

if(USE_ALLOW_NEGATIVE_SPECTRAL_WEIGHT)
    add_definitions(-D ALLOW_NEGATIVE_SPECTRAL_WEIGHT=1)
endif()

if(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF)
    add_definitions(-D USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF=1)
endif()

# Enable GPU support
set(GPU_BACKEND "none" CACHE STRING "Enable gpu accleration: cuda, hip, sycl, none (default: none)")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS cuda hip sycl none)
if (USE_BLAS AND "${GPU_BACKEND}" STREQUAL "none")
    message(FATAL_ERROR "USE_BLAS requires GPU_BACKEND to be set to cuda, hip, or sycl.")
endif()
if (USE_BLAS)
    add_definitions(-D USE_BLAS=1)
endif()
if (NOT "${GPU_BACKEND}" STREQUAL "none")
    add_definitions(-D USE_GPU=1)
    if ("${GPU_BACKEND}" STREQUAL "cuda")
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        add_definitions(-D USE_CUDA=1)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 70)
        endif()
        message("CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}.")
        include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    elseif("${GPU_BACKEND}" STREQUAL "hip")
        find_path(HIP_RUNTIME_INCLUDE_DIR
            NAMES hip/hip_runtime.h
            HINTS "$ENV{HIP_PATH}/include" "$ENV{ROCM_PATH}/include" /opt/rocm/include)
        if(NOT HIP_RUNTIME_INCLUDE_DIR)
            message(FATAL_ERROR "GPU_BACKEND=hip requires HIP headers. Set HIP_PATH, ROCM_PATH, or CMAKE_PREFIX_PATH.")
        endif()
        include_directories("${HIP_RUNTIME_INCLUDE_DIR}")
        add_definitions(-D USE_HIP=1)
    elseif("${GPU_BACKEND}" STREQUAL "sycl")
        add_definitions(-D USE_SYCL=1)
    else()
        message(FATAL_ERROR "Unknown GPU implementation '${GPU_BACKEND}',"
            " please select from 'cuda, hip, sycl, none'.")
    endif()
endif()

# Set number of threads per block
set(GPU_BLOCK_SIZE "1024" CACHE STRING "Set gpu block size (default: 1024)")
add_definitions(-D GPU_BLOCK_SIZE=${GPU_BLOCK_SIZE})

# Set sub-group size (warpsize/wavefront/SIMD lanes/etc.)
set(SUB_GROUP_SIZE "32" CACHE STRING "Set sub-group size (default: 32)")
add_definitions(-D SUB_GROUP_SIZE=${SUB_GROUP_SIZE})

# Set number of GPU streams
set(MAX_GPU_STREAMS "1" CACHE STRING "Set number of gpu streams (default: 1)")
add_definitions(-D MAX_GPU_STREAMS=${MAX_GPU_STREAMS})

# Set dynamic structure factor model
set(DEAC_MODEL "standard" CACHE STRING "Set deac-cpp model: standard, hyperbolic, normalization (default: standard)")
set_property(CACHE DEAC_MODEL PROPERTY STRINGS standard hyperbolic normalization)
if ("${DEAC_MODEL}" STREQUAL "standard")
    add_definitions(-D USE_STANDARD_MODEL=1)
elseif("${DEAC_MODEL}" STREQUAL "hyperbolic")
    add_definitions(-D USE_HYPERBOLIC_MODEL=1)
elseif("${DEAC_MODEL}" STREQUAL "normalization")
    add_definitions(-D USE_NORMALIZATION_MODEL=1)
else()
    message(FATAL_ERROR "Unknown deac-cpp model '${DEAC_MODEL}',"
        " please select from 'standard, hyperbolic, normalization'.")
endif()

if (SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION)
    add_definitions(-D SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION)
endif()

# Set default build flags
if("${GPU_BACKEND}" STREQUAL "sycl")
    if(NOT DEFINED SYCL_FLAGS)
        message(FATAL_ERROR "Please set SYCL_FLAGS CMake variable for target architecture and device,"
            " e.g. to target an NVIDIA A100 GPU:"
            "     '-DSYCL_FLAGS=\"-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80\"'")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS} -march=native -Wall -fno-math-errno -O3")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -fno-math-errno -O3")
endif()

# Determine executable name
set(exe deac.e)
if(CMAKE_BUILD_TYPE MATCHES ZeroTDebug)
    set(exe deac-zTd.e)
elseif(CMAKE_BUILD_TYPE MATCHES ZeroT)
    set(exe deac-zT.e)
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
    set(exe deacd.e)
endif()

# Generate a dedicated identity object and matching build-tree receipt.  Its
# symbolic refresh dependency deliberately recompiles this tiny object and
# relinks the executable on every ordinary build, avoiding timestamp races.
include("${CMAKE_SOURCE_DIR}/cmake/DeacBuildIdentity.cmake")
deac_add_build_identity(
    SOURCE_ROOT "${CMAKE_SOURCE_DIR}/.."
    GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated"
    IDENTITY_NAME deac
    RECEIPT "${CMAKE_CURRENT_BINARY_DIR}/deac-build-identity.json")

# Find source files ( better to list explicitly https://stackoverflow.com/questions/1027247/specify-source-files-globally-with-glob )
file(GLOB DEAC_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM DEAC_SRC "${DEAC_BUILD_IDENTITY_IMPLEMENTATION_SOURCE}")
if ("${GPU_BACKEND}" STREQUAL "cuda")
    file( GLOB DEAC_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu )
    add_executable(${exe}
        ${DEAC_SRC}
        ${DEAC_CUDA})
    set_target_properties( ${exe} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    #if(APPLE)
    #  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
    #  # so that the static cuda runtime can find it at runtime.
    #  set_property(TARGET ${exe} 
    #               PROPERTY
    #               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    #endif()
else()
    add_executable(${exe} ${DEAC_SRC})
endif()
set(DEAC_HIPBLAS_CONSUMER_TARGETS ${exe})

deac_target_add_build_identity(${exe})

# Define headers for target
target_include_directories(${exe} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE
        src)

if(USE_BLAS AND "${GPU_BACKEND}" STREQUAL "cuda")
    target_link_libraries(${exe} CUDA::cublas)
endif()

if(USE_BLAS AND "${GPU_BACKEND}" STREQUAL "sycl")
    set(MKLROOT "$ENV{MKLROOT}" CACHE PATH "oneMKL installation root")
    if(NOT MKLROOT)
        message(FATAL_ERROR "USE_BLAS with GPU_BACKEND=sycl requires MKLROOT to be set.")
    endif()
    target_include_directories(${exe} PRIVATE "${MKLROOT}/include")
    find_library(MKL_SYCL_BLAS_LIBRARY NAMES mkl_sycl_blas PATHS "${MKLROOT}/lib" NO_DEFAULT_PATH)
    find_library(MKL_SYCL_LIBRARY NAMES mkl_sycl PATHS "${MKLROOT}/lib" NO_DEFAULT_PATH)
    find_library(MKL_INTEL_ILP64_LIBRARY NAMES mkl_intel_ilp64 PATHS "${MKLROOT}/lib" NO_DEFAULT_PATH)
    find_library(MKL_SEQUENTIAL_LIBRARY NAMES mkl_sequential PATHS "${MKLROOT}/lib" NO_DEFAULT_PATH)
    find_library(MKL_CORE_LIBRARY NAMES mkl_core PATHS "${MKLROOT}/lib" NO_DEFAULT_PATH)
    foreach(mkl_lib_var
            MKL_SYCL_BLAS_LIBRARY
            MKL_SYCL_LIBRARY
            MKL_INTEL_ILP64_LIBRARY
            MKL_SEQUENTIAL_LIBRARY
            MKL_CORE_LIBRARY)
        if(NOT ${mkl_lib_var})
            message(FATAL_ERROR "Could not find required oneMKL library ${mkl_lib_var} under ${MKLROOT}/lib.")
        endif()
    endforeach()
    target_link_libraries(${exe}
        ${MKL_SYCL_BLAS_LIBRARY}
        ${MKL_SYCL_LIBRARY}
        ${MKL_INTEL_ILP64_LIBRARY}
        ${MKL_SEQUENTIAL_LIBRARY}
        ${MKL_CORE_LIBRARY}
        pthread
        m
        dl)
endif()

# Set debug flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEAC_DEBUG")

# Add ZeroT build mode (zero temperature)
set(CMAKE_CXX_FLAGS_ZEROT "${CMAKE_CXX_FLAGS} -DZEROT")
set(CMAKE_CXX_FLAGS_ZEROT "${CMAKE_CXX_FLAGS_ZEROT}" CACHE STRING
    "Flags used by the C++ compiler during ZeroT builds."
    FORCE )
set(CMAKE_C_FLAGS_ZEROT "" CACHE STRING
    "Flags used by the C compiler during ZeroT builds."
    FORCE )
set(CMAKE_EXE_LINKER_FLAGS_ZEROT
    "${CMAKE_EXE_LINKER_FLAGS}" CACHE STRING
    "Flags used for linking binaries during ZeroT builds."
    FORCE )
set(CMAKE_SHARED_LINKER_FLAGS_ZEROT
    "${CMAKE_SHARED_LINKER_FLAGS}" CACHE STRING
    "Flags used by the shared libraries linker during ZeroT builds."
    FORCE )

# Add ZeroTDebug build mode (zero temperature debug)
set(CMAKE_CXX_FLAGS_ZEROTDEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DZEROT")
set(CMAKE_CXX_FLAGS_ZEROTDEBUG "${CMAKE_CXX_FLAGS_ZEROTDEBUG}" CACHE STRING
    "Flags used by the C++ compiler during ZeroTDebug builds."
    FORCE )
set(CMAKE_C_FLAGS_ZEROTDEBUG "${CMAKE_C_FLAGS_DEBUG}" CACHE STRING
    "Flags used by the C compiler during ZeroTDebug builds."
    FORCE )
set(CMAKE_EXE_LINKER_FLAGS_ZEROTDEBUG
    "${CMAKE_EXE_LINKER_FLAGS_DEBUG}" CACHE STRING
    "Flags used for linking binaries during ZeroTDebug builds."
    FORCE )
set(CMAKE_SHARED_LINKER_FLAGS_ZEROTDEBUG
    "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" CACHE STRING
    "Flags used by the shared libraries linker during ZeroTDebug builds."
    FORCE )

mark_as_advanced(
    CMAKE_CXX_FLAGS_ZEROT
    CMAKE_C_FLAGS_ZEROT
    CMAKE_EXE_LINKER_FLAGS_ZEROT
    CMAKE_SHARED_LINKER_FLAGS_ZEROT
    CMAKE_CXX_FLAGS_ZEROTDEBUG
    CMAKE_C_FLAGS_ZEROTDEBUG
    CMAKE_EXE_LINKER_FLAGS_ZEROTDEBUG
    CMAKE_SHARED_LINKER_FLAGS_ZEROTDEBUG )

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE None
       CACHE STRING "Choose the type of build : None Debug Release ZeroT ZeroTDebug."
       FORCE)
endif()

# Link filesystem library -lstdc++fs for old compilers
if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    message("Linking filesystem libraries -lstdc++fs for older compilers")
    target_link_libraries(${exe} stdc++fs)
endif()

# Embed a target-specific attribution receipt generated from CMake's final
# codemodel, toolchain, and cache replies.  The adjacent JSON is diagnostic;
# the executable's --build-receipt response is authoritative.
include("${CMAKE_SOURCE_DIR}/cmake/DeacBuildReceipt.cmake")
set(DEAC_BUILD_RECEIPT_CACHE_KEYS
    BUILD_SHARED_LIBS
    CMAKE_AR
    CMAKE_BUILD_TYPE
    CMAKE_CONFIGURATION_TYPES
    CMAKE_CUDA_ARCHITECTURES
    CMAKE_CUDA_COMPILER
    CMAKE_CUDA_FLAGS
    CMAKE_CXX_COMPILER
    CMAKE_CXX_COMPILER_LAUNCHER
    CMAKE_CXX_FLAGS
    CMAKE_EXE_LINKER_FLAGS
    CMAKE_GENERATOR
    CMAKE_GENERATOR_INSTANCE
    CMAKE_GENERATOR_PLATFORM
    CMAKE_GENERATOR_TOOLSET
    CMAKE_HIP_ARCHITECTURES
    CMAKE_HOME_DIRECTORY
    CMAKE_INTERPROCEDURAL_OPTIMIZATION
    CMAKE_LINKER
    CMAKE_MAKE_PROGRAM
    CMAKE_PREFIX_PATH
    CMAKE_RANLIB
    CMAKE_SYSROOT
    CMAKE_SYSROOT_COMPILE
    CMAKE_SYSROOT_LINK
    CMAKE_TOOLCHAIN_FILE
    DEAC_MODEL
    GPU_BACKEND
    GPU_BLOCK_SIZE
    MAX_GPU_STREAMS
    MKLROOT
    SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION
    STATIC
    SUB_GROUP_SIZE
    SYCL_FLAGS
    USE_ALLOW_NEGATIVE_SPECTRAL_WEIGHT
    USE_BLAS
    USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF
    USE_SYCL_MATMUL_WA)
foreach(_configuration
        DEBUG RELEASE MINSIZEREL RELWITHDEBINFO ZEROT ZEROTDEBUG)
    list(APPEND DEAC_BUILD_RECEIPT_CACHE_KEYS
        "CMAKE_CXX_FLAGS_${_configuration}"
        "CMAKE_EXE_LINKER_FLAGS_${_configuration}")
endforeach()
deac_target_add_build_receipt(${exe}
    SOURCE_ROOT "${CMAKE_SOURCE_DIR}/.."
    GENERATED_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/receipt"
    IDENTITY_NAME deac
    RECEIPT
        "${CMAKE_CURRENT_BINARY_DIR}/receipt/$<CONFIG>/deac-build-receipt.json"
    BACKEND "${GPU_BACKEND}"
    CACHE_KEYS ${DEAC_BUILD_RECEIPT_CACHE_KEYS}
    DEPENDENCY_TARGETS ${DEAC_BUILD_IDENTITY_OBJECT_TARGET})

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS ${exe} EXPORT ${exe}Config
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${exe}Config DESTINATION share/${exe}/cmake)

# This makes the project importable from the build directory
export(TARGETS ${exe} FILE ${exe}Config.cmake)

# Unit tests
set(DEAC_HIPBLAS_CONFIGURE_TEST_RUNNER
    "${CMAKE_SOURCE_DIR}/../test/cmake/RunHipblasConfigureTest.cmake")
set(DEAC_HIPBLAS_CONFIGURE_TEST_SOURCE
    "${CMAKE_SOURCE_DIR}/../test/cmake/hipblas_link_contract")
function(deac_add_hipblas_configure_test test_mode expect_success)
    add_test(
        NAME deac_cmake_hipblas_${test_mode}
        COMMAND
            ${CMAKE_COMMAND}
            "-DDEAC_TEST_CMAKE_COMMAND:FILEPATH=${CMAKE_COMMAND}"
            "-DDEAC_TEST_SOURCE_DIR:PATH=${DEAC_HIPBLAS_CONFIGURE_TEST_SOURCE}"
            "-DDEAC_TEST_BINARY_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/hipblas-configure/${test_mode}"
            "-DDEAC_TEST_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
            "-DDEAC_TEST_GENERATOR:STRING=${CMAKE_GENERATOR}"
            "-DDEAC_TEST_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}"
            "-DDEAC_TEST_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}"
            "-DDEAC_TEST_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}"
            "-DDEAC_TEST_GENERATOR_INSTANCE:STRING=${CMAKE_GENERATOR_INSTANCE}"
            "-DDEAC_TEST_HIPBLAS_MODULE:FILEPATH=${CMAKE_SOURCE_DIR}/cmake/DeacHipblas.cmake"
            "-DDEAC_TEST_MODE:STRING=${test_mode}"
            "-DDEAC_TEST_EXPECT_SUCCESS:BOOL=${expect_success}"
            -P "${DEAC_HIPBLAS_CONFIGURE_TEST_RUNNER}")
    set_tests_properties(deac_cmake_hipblas_${test_mode} PROPERTIES
        LABELS "cmake;hipblas")
endfunction()

deac_add_hipblas_configure_test(imported_target TRUE)
deac_add_hipblas_configure_test(compatibility_library TRUE)
deac_add_hipblas_configure_test(missing_package FALSE)
deac_add_hipblas_configure_test(header_only FALSE)
deac_add_hipblas_configure_test(blas_off TRUE)
deac_add_hipblas_configure_test(non_hip TRUE)
add_test(
    NAME deac_cmake_hipblas_project_targets
    COMMAND
        ${CMAKE_COMMAND}
        "-DDEAC_TEST_CMAKE_COMMAND:FILEPATH=${CMAKE_COMMAND}"
        "-DDEAC_TEST_PROJECT_SOURCE_DIR:PATH=${CMAKE_SOURCE_DIR}"
        "-DDEAC_TEST_BINARY_DIR:PATH=${CMAKE_CURRENT_BINARY_DIR}/hipblas-configure/project_targets"
        "-DDEAC_TEST_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
        "-DDEAC_TEST_GENERATOR:STRING=${CMAKE_GENERATOR}"
        "-DDEAC_TEST_MAKE_PROGRAM:FILEPATH=${CMAKE_MAKE_PROGRAM}"
        "-DDEAC_TEST_GENERATOR_PLATFORM:STRING=${CMAKE_GENERATOR_PLATFORM}"
        "-DDEAC_TEST_GENERATOR_TOOLSET:STRING=${CMAKE_GENERATOR_TOOLSET}"
        "-DDEAC_TEST_GENERATOR_INSTANCE:STRING=${CMAKE_GENERATOR_INSTANCE}"
        "-DDEAC_TEST_MOCK_INCLUDE_DIR:PATH=${DEAC_HIPBLAS_CONFIGURE_TEST_SOURCE}/mock-include"
        "-DDEAC_TEST_MOCK_PACKAGE_DIR:PATH=${DEAC_HIPBLAS_CONFIGURE_TEST_SOURCE}/mock-package"
        -P "${CMAKE_SOURCE_DIR}/../test/cmake/RunHipblasProjectConfigureTest.cmake")
set_tests_properties(deac_cmake_hipblas_project_targets PROPERTIES
    LABELS "cmake;hipblas")

add_executable(deac_trapezoidal_weights_test
    ${CMAKE_SOURCE_DIR}/../test/trapezoidal_weights_test.cpp)
target_include_directories(deac_trapezoidal_weights_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(
    NAME deac_unit_trapezoidal_weights
    COMMAND deac_trapezoidal_weights_test)

add_executable(deac_evolution_controls_test
    ${CMAKE_SOURCE_DIR}/../test/evolution_controls_test.cpp)
target_include_directories(deac_evolution_controls_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(
    NAME deac_unit_evolution_controls
    COMMAND deac_evolution_controls_test)

add_executable(deac_normalization_test
    ${CMAKE_SOURCE_DIR}/../test/normalization_test.cpp)
target_include_directories(deac_normalization_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(
    NAME deac_unit_normalization
    COMMAND deac_normalization_test)

find_package(Python3 COMPONENTS Interpreter)
foreach(status_mode IN ITEMS debug ndebug)
    set(status_target deac_gpu_status_${status_mode}_test)
    add_executable(${status_target}
        ${CMAKE_SOURCE_DIR}/../test/gpu_status_test.cpp)
    set_target_properties(${status_target} PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE)
    target_include_directories(${status_target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(status_mode STREQUAL "ndebug")
        target_compile_definitions(${status_target} PRIVATE
            DEAC_TEST_WITH_NDEBUG=1)
    endif()
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME deac_unit_gpu_status_${status_mode}
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/../test/gpu_status_driver_test.py
                --exe $<TARGET_FILE:${status_target}>)
    endif()
endforeach()

if(NOT "${GPU_BACKEND}" STREQUAL "none")
    set(DEAC_GPU_NORMALIZATION_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/../test/gpu_normalization_test.cpp)
    if("${GPU_BACKEND}" STREQUAL "cuda")
        list(APPEND DEAC_GPU_NORMALIZATION_TEST_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/deac_gpu.cu)
    endif()
    add_executable(deac_gpu_normalization_test
        ${DEAC_GPU_NORMALIZATION_TEST_SOURCES})
    list(APPEND DEAC_HIPBLAS_CONSUMER_TARGETS
        deac_gpu_normalization_test)
    target_include_directories(deac_gpu_normalization_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if("${GPU_BACKEND}" STREQUAL "cuda")
        set_target_properties(deac_gpu_normalization_test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    endif()
    if(USE_BLAS AND "${GPU_BACKEND}" STREQUAL "cuda")
        target_link_libraries(deac_gpu_normalization_test CUDA::cublas)
    endif()
    if(USE_BLAS AND "${GPU_BACKEND}" STREQUAL "sycl")
        target_link_libraries(deac_gpu_normalization_test
            ${MKL_SYCL_BLAS_LIBRARY}
            ${MKL_SYCL_LIBRARY}
            ${MKL_INTEL_ILP64_LIBRARY}
            ${MKL_SEQUENTIAL_LIBRARY}
            ${MKL_CORE_LIBRARY}
            pthread
            m
            dl)
    endif()
    add_test(
        NAME deac_unit_gpu_normalization
        COMMAND deac_gpu_normalization_test)

    set(DEAC_GPU_FITNESS_TEST_SOURCES
        ${CMAKE_SOURCE_DIR}/../test/gpu_fitness_test.cpp)
    if("${GPU_BACKEND}" STREQUAL "cuda")
        list(APPEND DEAC_GPU_FITNESS_TEST_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/deac_gpu.cu)
    endif()
    add_executable(deac_gpu_fitness_test
        ${DEAC_GPU_FITNESS_TEST_SOURCES})
    list(APPEND DEAC_HIPBLAS_CONSUMER_TARGETS
        deac_gpu_fitness_test)
    target_include_directories(deac_gpu_fitness_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if("${GPU_BACKEND}" STREQUAL "cuda")
        set_target_properties(deac_gpu_fitness_test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    endif()
    if(USE_BLAS AND "${GPU_BACKEND}" STREQUAL "cuda")
        target_link_libraries(deac_gpu_fitness_test CUDA::cublas)
    endif()
    if(USE_BLAS AND "${GPU_BACKEND}" STREQUAL "sycl")
        target_link_libraries(deac_gpu_fitness_test
            ${MKL_SYCL_BLAS_LIBRARY}
            ${MKL_SYCL_LIBRARY}
            ${MKL_INTEL_ILP64_LIBRARY}
            ${MKL_SEQUENTIAL_LIBRARY}
            ${MKL_CORE_LIBRARY}
            pthread
            m
            dl)
    endif()
    add_test(
        NAME deac_unit_gpu_fitness
        COMMAND deac_gpu_fitness_test)
endif()

add_executable(deac_moment_fitness_test
    ${CMAKE_SOURCE_DIR}/../test/moment_fitness_test.cpp)
target_include_directories(deac_moment_fitness_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(
    NAME deac_unit_moment_fitness
    COMMAND deac_moment_fitness_test)

add_executable(deac_population_projection_test
    ${CMAKE_SOURCE_DIR}/../test/population_projection_test.cpp)
target_include_directories(deac_population_projection_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(
    NAME deac_unit_population_projection
    COMMAND deac_population_projection_test)

add_executable(deac_trial_population_test
    ${CMAKE_SOURCE_DIR}/../test/trial_population_test.cpp)
target_include_directories(deac_trial_population_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(
    NAME deac_unit_trial_population
    COMMAND deac_trial_population_test)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM"
        AND NOT GPU_BACKEND STREQUAL "cuda"
        AND NOT GPU_BACKEND STREQUAL "hip")
    add_executable(deac_trial_population_no_contract_test
        ${CMAKE_SOURCE_DIR}/../test/trial_population_test.cpp)
    target_include_directories(deac_trial_population_no_contract_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(deac_trial_population_no_contract_test PRIVATE
        -ffp-contract=off)
    add_test(
        NAME deac_unit_trial_population_no_contract
        COMMAND deac_trial_population_no_contract_test)
endif()

add_executable(deac_result_io_test
    ${CMAKE_SOURCE_DIR}/../test/result_io_test.cpp)
target_include_directories(deac_result_io_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(
    NAME deac_unit_result_io
    COMMAND deac_result_io_test)

add_executable(deac_zero_temperature_kernel_test
    ${CMAKE_SOURCE_DIR}/../test/zero_temperature_kernel_test.cpp)
target_include_directories(deac_zero_temperature_kernel_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(
    NAME deac_unit_zero_temperature_kernel
    COMMAND deac_zero_temperature_kernel_test)

# A separate test-only solver binary forces every evolved normalization
# denominator to zero.  Keeping the hook out of the installed solver makes the
# complete-loop rejection regression deterministic without adding a public
# option or runtime branch.
if(NOT USE_ALLOW_NEGATIVE_SPECTRAL_WEIGHT)
    if("${GPU_BACKEND}" STREQUAL "cuda")
        add_executable(deac_invalid_normalization_evolution_test_exe
            ${DEAC_SRC} ${DEAC_CUDA})
        set_target_properties(
            deac_invalid_normalization_evolution_test_exe PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    else()
        add_executable(deac_invalid_normalization_evolution_test_exe
            ${DEAC_SRC})
    endif()
    list(APPEND DEAC_HIPBLAS_CONSUMER_TARGETS
        deac_invalid_normalization_evolution_test_exe)
    deac_target_add_build_identity(
        deac_invalid_normalization_evolution_test_exe)
    target_compile_definitions(
        deac_invalid_normalization_evolution_test_exe PRIVATE
        DEAC_TEST_FORCE_INVALID_NORMALIZATION_TRIALS=1)
    if(NOT "${GPU_BACKEND}" STREQUAL "none")
        target_compile_definitions(
            deac_invalid_normalization_evolution_test_exe PRIVATE
            DEAC_TEST_POISON_GPU_FITNESS=1)
    endif()
    target_include_directories(
        deac_invalid_normalization_evolution_test_exe PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE src)
    if(USE_BLAS AND "${GPU_BACKEND}" STREQUAL "cuda")
        target_link_libraries(
            deac_invalid_normalization_evolution_test_exe CUDA::cublas)
    endif()
    if(USE_BLAS AND "${GPU_BACKEND}" STREQUAL "sycl")
        target_include_directories(
            deac_invalid_normalization_evolution_test_exe PRIVATE
            "${MKLROOT}/include")
        target_link_libraries(
            deac_invalid_normalization_evolution_test_exe
            ${MKL_SYCL_BLAS_LIBRARY}
            ${MKL_SYCL_LIBRARY}
            ${MKL_INTEL_ILP64_LIBRARY}
            ${MKL_SEQUENTIAL_LIBRARY}
            ${MKL_CORE_LIBRARY}
            pthread
            m
            dl)
    endif()
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(
            deac_invalid_normalization_evolution_test_exe stdc++fs)
    endif()
endif()

# The HIP+BLAS package, includes, and link dependency are represented by one
# target-scoped contract shared by every target that compiles the GPU solver
# implementation.  The helper is a no-op for every other configuration.
deac_target_link_hipblas(${DEAC_HIPBLAS_CONSUMER_TARGETS})

if(Python3_Interpreter_FOUND)
    add_test(
        NAME deac_gpu_status_source_policy
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/../test/gpu_status_policy_test.py
            --source-root ${CMAKE_SOURCE_DIR}/..)
    set(DEAC_SMOKE_TEST "${CMAKE_SOURCE_DIR}/../test/deac_smoke_test.py")
    set(DEAC_PARITY_TEST "${CMAKE_SOURCE_DIR}/../test/deac_parity_test.py")
    set(DEAC_BUILD_IDENTITY_TEST
        "${CMAKE_SOURCE_DIR}/../test/deac_build_identity_test.py")
    set(DEAC_BUILD_RECEIPT_TEST
        "${CMAKE_SOURCE_DIR}/../test/deac_build_receipt_test.py")
    set(DEAC_BUILD_RECEIPT_FIXTURE_TEST
        "${CMAKE_SOURCE_DIR}/../test/deac_build_receipt_fixture_test.py")
    find_program(DEAC_NINJA_EXECUTABLE NAMES ninja)
    set(DEAC_SMOKE_VARIANT_ARGS)
    if(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF)
        list(APPEND DEAC_SMOKE_VARIANT_ARGS --detailed-balance)
    endif()
    if(CMAKE_BUILD_TYPE MATCHES "^ZeroT")
        list(APPEND DEAC_SMOKE_VARIANT_ARGS --zero-temperature)
    endif()
    set(DEAC_SMOKE_CASES
            help
            version
            bad_spectra
            default
            evolution_control_lower_boundaries
            evolution_control_upper_boundaries
            first_moment
            third_moment
            track_stats
            valid_nonuniform_frequency
            bad_isf_byte_length
            empty_isf
            uneven_isf_arrays
            too_few_timeslices
            nonfinite_isf
            normalize_zero_target
            normalize_negative_target
            normalize_subnormal_target
            normalize_nonfinite_target
            missing_isf
            positive_isf_single_particle
            bad_third_moment_error
            bad_crossover_probability
            bad_self_adapting_crossover_probability
            bad_differential_weight
            bad_self_adapting_differential_weight_probability
            bad_stop_minimum_fitness
            inactive_options_rejected
            too_few_generations
            too_small_population
            too_small_genome
            bad_omega_max
            missing_frequency
            short_frequency_file
            nonfinite_frequency
            negative_frequency
            duplicate_frequency
            all_equal_frequency
            unsorted_frequency
            nested_output
            save_directory_is_file
            unwritable_save_directory
            log_destination_is_directory
            result_destination_is_directory)
    if(USE_ALLOW_NEGATIVE_SPECTRAL_WEIGHT)
        # Basic malformed-target validation still takes precedence above.  A
        # valid target reaches this build-mode incompatibility check.
        list(APPEND DEAC_SMOKE_CASES normalize_signed_weights)
    else()
        list(APPEND DEAC_SMOKE_CASES
            normalize
            normalize_large_population
            normalize_unrepresentable_scale)
    endif()
    if(USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF)
        list(APPEND DEAC_SMOKE_CASES negative_first_moment)
    else()
        list(APPEND DEAC_SMOKE_CASES unsupported_negative_first_moment)
    endif()
    foreach(test_case IN LISTS DEAC_SMOKE_CASES)
        add_test(
            NAME deac_smoke_${test_case}
            COMMAND
                ${Python3_EXECUTABLE}
                ${DEAC_SMOKE_TEST}
                --exe $<TARGET_FILE:${exe}>
                --workdir ${CMAKE_CURRENT_BINARY_DIR}/smoke/${test_case}
                --case ${test_case}
                --expected-version ${DEAC_BUILD_IDENTITY_SEMANTIC_VERSION}
                ${DEAC_SMOKE_VARIANT_ARGS}
        )
    endforeach()
    if(NOT USE_ALLOW_NEGATIVE_SPECTRAL_WEIGHT)
        set(DEAC_INVALID_NORMALIZATION_TEST_ARGS)
        if(NOT "${GPU_BACKEND}" STREQUAL "none")
            list(APPEND DEAC_INVALID_NORMALIZATION_TEST_ARGS
                --expect-poisoned-gpu-fitness)
        endif()
        add_test(
            NAME deac_normalization_evolution
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/../test/deac_normalization_evolution_test.py
                --exe $<TARGET_FILE:${exe}>
                --workdir ${CMAKE_CURRENT_BINARY_DIR}/normalization-evolution
                ${DEAC_SMOKE_VARIANT_ARGS}
        )
        add_test(
            NAME deac_invalid_normalization_evolution
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/../test/deac_invalid_normalization_evolution_test.py
                --reference-exe $<TARGET_FILE:${exe}>
                --forced-exe $<TARGET_FILE:deac_invalid_normalization_evolution_test_exe>
                --workdir ${CMAKE_CURRENT_BINARY_DIR}/invalid-normalization-evolution
                ${DEAC_SMOKE_VARIANT_ARGS}
                ${DEAC_INVALID_NORMALIZATION_TEST_ARGS}
        )
    endif()

    set(DEAC_BUILD_IDENTITY_GIT_ARGS)
    if(Git_FOUND)
        list(APPEND DEAC_BUILD_IDENTITY_GIT_ARGS
            --git "${GIT_EXECUTABLE}")
    endif()
    set(DEAC_BUILD_IDENTITY_CONFIG_ARGS)
    if(CMAKE_CONFIGURATION_TYPES)
        list(APPEND DEAC_BUILD_IDENTITY_CONFIG_ARGS
            --build-config $<CONFIG>)
    endif()
    add_test(
        NAME deac_build_identity
        COMMAND
            ${Python3_EXECUTABLE}
            ${DEAC_BUILD_IDENTITY_TEST}
            --exe $<TARGET_FILE:${exe}>
            --receipt ${DEAC_BUILD_IDENTITY_RECEIPT}
            --source-root ${CMAKE_SOURCE_DIR}/..
            --workdir ${CMAKE_CURRENT_BINARY_DIR}/build-identity-test
            --identity-module
                ${CMAKE_SOURCE_DIR}/cmake/DeacBuildIdentity.cmake
            --cmake ${CMAKE_COMMAND}
            --cxx-compiler ${CMAKE_CXX_COMPILER}
            --generator ${CMAKE_GENERATOR}
            --make-program ${CMAKE_MAKE_PROGRAM}
            ${DEAC_BUILD_IDENTITY_CONFIG_ARGS}
            ${DEAC_BUILD_IDENTITY_GIT_ARGS}
    )
    add_test(
        NAME deac_build_receipt
        COMMAND
            ${Python3_EXECUTABLE}
            ${DEAC_BUILD_RECEIPT_TEST}
            --exe $<TARGET_FILE:${exe}>
            --receipt ${DEAC_BUILD_RECEIPT}
            --build-dir ${CMAKE_BINARY_DIR}
            --source-dir ${CMAKE_SOURCE_DIR}
            --workdir ${CMAKE_CURRENT_BINARY_DIR}/build-receipt-test
            --cmake ${CMAKE_COMMAND}
            --expected-backend ${GPU_BACKEND}
            --expected-target ${exe}
            --expected-dependency-target ${DEAC_BUILD_IDENTITY_OBJECT_TARGET}
            ${DEAC_BUILD_IDENTITY_CONFIG_ARGS}
    )
    if(DEAC_NINJA_EXECUTABLE)
        add_test(
            NAME deac_build_receipt_fixture
            COMMAND
                ${Python3_EXECUTABLE}
                ${DEAC_BUILD_RECEIPT_FIXTURE_TEST}
                --cmake ${CMAKE_COMMAND}
                --ninja ${DEAC_NINJA_EXECUTABLE}
                --solver-source-root ${CMAKE_SOURCE_DIR}
                --cxx-compiler ${CMAKE_CXX_COMPILER}
                --workdir
                    ${CMAKE_CURRENT_BINARY_DIR}/build-receipt-fixture-test
        )
    endif()
    add_test(
        NAME deac_first_moment_fitness
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/../test/deac_first_moment_fitness_test.py
            --exe $<TARGET_FILE:${exe}>
            --workdir ${CMAKE_CURRENT_BINARY_DIR}/first-moment-fitness
            ${DEAC_SMOKE_VARIANT_ARGS}
    )
    set(DEAC_PARITY_REFERENCE_EXE "" CACHE FILEPATH "Reference deac executable for backend parity tests.")
    if(NOT "${DEAC_PARITY_REFERENCE_EXE}" STREQUAL "")
        add_test(
            NAME deac_parity_reference
            COMMAND
                ${Python3_EXECUTABLE}
                ${DEAC_PARITY_TEST}
                --reference-exe ${DEAC_PARITY_REFERENCE_EXE}
                --candidate-exe $<TARGET_FILE:${exe}>
                --workdir ${CMAKE_CURRENT_BINARY_DIR}/parity/reference
                ${DEAC_SMOKE_VARIANT_ARGS}
        )
    endif()
else()
    message(WARNING "Python3 interpreter not found; deac smoke tests will not be registered.")
endif()
