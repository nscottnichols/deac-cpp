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

# Check for GPU acceleration
if (GPU_BLOCK_SIZE)
    add_definitions(-D GPU_BLOCK_SIZE=${GPU_BLOCK_SIZE})
    if (NOT MAX_GPU_STREAMS)
        set(MAX_GPU_STREAMS 1)
    endif()
    add_definitions(-D MAX_GPU_STREAMS=${MAX_GPU_STREAMS})
endif()

# Set default build flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -fno-math-errno -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-math-errno -O3")

# Determine executable name
set(exe deac.e)
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(exe daecd.e)
elseif(CMAKE_BUILD_TYPE MATCHES ZeroT)
    set(exe deac_zT.e)
elseif(CMAKE_BUILD_TYPE MATCHES ZeroTDebug)
    set(exe deac_zTd.e)
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
