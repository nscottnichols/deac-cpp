cmake_minimum_required(VERSION 3.18.0)

project(deac.e LANGUAGES CXX)

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
        if(USE_BLAS)
            find_path(HIPBLAS_INCLUDE_DIR
                NAMES hipblas/hipblas.h
                HINTS "$ENV{HIP_PATH}/include" "$ENV{ROCM_PATH}/include" /opt/rocm/include)
            if(NOT HIPBLAS_INCLUDE_DIR)
                message(FATAL_ERROR "USE_BLAS with GPU_BACKEND=hip requires hipBLAS headers.")
            endif()
            include_directories("${HIPBLAS_INCLUDE_DIR}")
        endif()
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

# Find source files ( better to list explicitly https://stackoverflow.com/questions/1027247/specify-source-files-globally-with-glob )
file( GLOB DEAC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp )
if ("${GPU_BACKEND}" STREQUAL "cuda")
    file( GLOB DEAC_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu )
    add_executable(${exe} ${DEAC_SRC} ${DEAC_CUDA})
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

# Define headers for target
target_include_directories(${exe} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)

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
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    set(DEAC_SMOKE_TEST "${CMAKE_SOURCE_DIR}/../test/deac_smoke_test.py")
    set(DEAC_PARITY_TEST "${CMAKE_SOURCE_DIR}/../test/deac_parity_test.py")
    foreach(test_case
            help
            bad_spectra
            default
            normalize
            first_moment
            track_stats
            bad_isf_byte_length
            empty_isf
            uneven_isf_arrays
            too_few_timeslices
            nonfinite_isf
            positive_isf_single_particle
            bad_third_moment_error
            too_few_generations
            too_small_population
            too_small_genome
            bad_omega_max
            short_frequency_file
            nonfinite_frequency
            negative_frequency
            unsorted_frequency)
        add_test(
            NAME deac_smoke_${test_case}
            COMMAND
                ${Python3_EXECUTABLE}
                ${DEAC_SMOKE_TEST}
                --exe $<TARGET_FILE:${exe}>
                --workdir ${CMAKE_CURRENT_BINARY_DIR}/smoke/${test_case}
                --case ${test_case}
        )
    endforeach()
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
        )
    endif()
else()
    message(WARNING "Python3 interpreter not found; deac smoke tests will not be registered.")
endif()
