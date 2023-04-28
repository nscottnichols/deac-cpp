cmake_minimum_required(VERSION 3.18.0)

project(deac.e LANGUAGES CXX)

# Set c++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_STANDARD_REQUIRED TRUE)

# Check for static build
if(STATIC)
    message("Static build specified, setting library suffixes to ${CMAKE_STATIC_LIBRARY_SUFFIX}.")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(BUILD_SHARED_LIBS OFF)
    list(APPEND CMAKE_EXE_LINKER_FLAGS "-static")
    list(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG "-static")
endif()

# Check for <span> flags
if (USE_TCB_SPAN)
    add_definitions(-D USE_TCB_SPAN=${USE_TCB_SPAN})
endif()

# Enable GPU support
set(GPU_BACKEND "none" CACHE STRING "Enable gpu accleration: cuda, hip, sycl, none (default: none)")
if (NOT ${GPU_BACKEND} STREQUAL "none")
    add_definitions(-D USE_GPU=1)
    if (${GPU_BACKEND} STREQUAL "cuda")
        enable_language(CUDA)
        add_definitions(-D USE_CUDA=1)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 70)
        endif()
        message("CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}.")
        include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    elseif(${GPU_BACKEND} STREQUAL "hip")
        add_definitions(-D USE_HIP=1)
    elseif(${GPU_BACKEND} STREQUAL "sycl")
        add_definitions(-D USE_SYCL=1)
    else()
        message(FATAL_ERROR "Unknown GPU implementation '${GPU_BACKEND}',"
            " please select from 'cuda, hip, sycl, none'.")
    endif()
endif()

# Set number of threads per block
set(GPU_BLOCK_SIZE "1024" CACHE STRING "Set number of gpu block size (default: 1024)")
add_definitions(-D GPU_BLOCK_SIZE=${GPU_BLOCK_SIZE})

# Set number of GPU streams
set(MAX_GPU_STREAMS "1" CACHE STRING "Set number of gpu streams (default: 1)")
add_definitions(-D MAX_GPU_STREAMS=${MAX_GPU_STREAMS})

# Set dynamic structure factor model
set(DEAC_MODEL "standard" CACHE STRING "Set deac-cpp model: standard, hyperbolic, normalization (default: standard)")
if (${DEAC_MODEL} STREQUAL "standard")
    add_definitions(-D USE_STANDARD_MODEL=1)
elseif(${DEAC_MODEL} STREQUAL "hyperbolic")
    add_definitions(-D USE_HYPERBOLIC_MODEL=1)
elseif(${DEAC_MODEL} STREQUAL "normalization")
    add_definitions(-D USE_NORMALIZATION_MODEL=1)
else()
    message(FATAL_ERROR "Unknown deac-cpp model '${DEAC_MODEL}',"
        " please select from 'standard, hyperbolic, normalization'.")
endif()

if (SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION)
    add_definitions(-D SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION)
endif()

# Set default build flags
if(${GPU_BACKEND} STREQUAL "sycl")
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
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(exe daecd.e)
elseif(CMAKE_BUILD_TYPE MATCHES ZeroT)
    set(exe deac-zT.e)
elseif(CMAKE_BUILD_TYPE MATCHES ZeroTDebug)
    set(exe deac-zTd.e)
endif()

# Find source files ( better to list explicitly https://stackoverflow.com/questions/1027247/specify-source-files-globally-with-glob )
file( GLOB DEAC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp )
if(USE_CUDA)
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
# We need to make some of these
