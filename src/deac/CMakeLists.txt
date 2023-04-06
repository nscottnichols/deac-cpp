cmake_minimum_required(VERSION 3.18.0)

if (USE_CUDA)
    project(deac.e LANGUAGES CXX CUDA)
    add_definitions(-D USE_CUDA=${USE_CUDA})
else()
    project(deac.e LANGUAGES CXX)
endif()

# Set c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_STANDARD_REQUIRED TRUE)
if (USE_CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70)
    endif()
    message("CUDA architectures set to ${CMAKE_CUDA_ARCHITECTURES}.")
endif()

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
set(GPU_BACKEND "none" CACHE STRING "Enable gpu accleration: cuda, hip, none (default: none)")
if (NOT ${GPU_BACKEND} STREQUAL "none")
    add_definitions(-D USE_GPU=1)
    if (${GPU_BACKEND} STREQUAL "cuda")
        add_definitions(-D USE_CUDA=1)
    elseif(${GPU_BACKEND} STREQUAL "hip")
        add_definitions(-D USE_HIP=1)
    else()
        message(FATAL_ERROR "Unknown GPU implementation '${GPU_BACKEND}',"
            " please select from 'cuda, hip, none'.")
    endif()
endif()

# Set number of threads per block
if (GPU_BLOCK_SIZE)
    add_definitions(-D GPU_BLOCK_SIZE=${GPU_BLOCK_SIZE})
    add_definitions(-D MAX_GPU_STREAMS=${MAX_GPU_STREAMS})
endif()

# Set number of GPU streams
if (MAX_GPU_STREAMS)
    add_definitions(-D MAX_GPU_STREAMS=${MAX_GPU_STREAMS})
endif()

# Set dynamic structure factor model
if ((NOT USE_HYPERBOLIC_MODEL) AND (NOT USE_STANDARD_MODEL) AND (NOT USE_NORMALIZATION_MODEL))
    set(USE_HYPERBOLIC_MODEL 1)
endif()

# Ensure only one model is selected
if (( USE_HYPERBOLIC_MODEL AND USE_STANDARD_MODEL ) OR 
    ( USE_HYPERBOLIC_MODEL AND USE_NORMALIZATION_MODEL ) OR
    ( USE_STANDARD_MODEL AND USE_NORMALIZATION_MODEL ))
    message( FATAL_ERROR "Please only set one model: USE_HYPERBOLIC_MODEL, USE_STANDARD_MODEL, USE_NORMALIZATION_MODEL" )
endif()

if (USE_HYPERBOLIC_MODEL)
    add_definitions(-D USE_HYPERBOLIC_MODEL)
endif()

if (USE_STANDARD_MODEL)
    add_definitions(-D USE_STANDARD_MODEL)
endif()

if (USE_NORMALIZATION_MODEL)
    add_definitions(-D USE_NORMALIZATION_MODEL)
endif()

if (SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION)
    add_definitions(-D SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION)
endif()

# Set default build flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -fno-math-errno -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-math-errno -O3")

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
