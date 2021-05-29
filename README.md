# deac-cpp
A C++ implementation of the Differential Evolution for Analytic Continuation (DEAC) algorithm which uses self adaptive differential evolution to reconstruct the dynamic structure factor from imaginary time density-density correlations.

## Quick installation
DEAC uses CMake for build, test and installation automation. For details on using CMake consult https://cmake.org/documentation/. In short, the following steps should work on UNIX-like systems:

  ```
  git clone git@github.com:nscottnichols/deac-cpp.git
  cd deac-cpp
  mkdir build
  cd build
  cmake ../src
  make
  sudo make install
  ```

On Windows try:

  ```
  git clone git@github.com:nscottnichols/deac-cpp.git
  cd deac-cpp
  md build
  cd build
  cmake ../src
  cmake --build . --config Release
  cmake --build . --target install
  ```

## Typical installation
As above, and with further details below, but you should consider using the following CMake options with the appropriate value instead of xxx :

- `-D USE_HYPERBOLIC_MODEL=1` use hyperbolic model for spectral function S'(ω)=2S(ω)e^(-βω/2)
- `-D USE_STANDARD_MODEL=1` use standard model for spectral function S'(ω)=S(ω)
- `-D USE_NORMALIZATION_MODEL=1` use normalization model for spectral function S'(ω)=S(ω)(1 + e^(-βω))
- `-D GPU_BLOCK_SIZE=xxx` equal to the maximum threadblock size and enables GPU acceleration (using [AMD's HIP language](https://github.com/ROCm-Developer-Tools/HIP))
- `-D MAX_GPU_STREAMS=xxx` equal to maximum number of concurrent streams on GPU device
- `-D USE_CUDA=1` use CUDA instead of HIP for GPU acceleration
- `-D CMAKE_CUDA_ARCHITECTURES=xxx` equal to CUDA device architecture if not properly detected by CMake
- `-D USE_TCB_SPAN=1` use [Tristan Brindle's span](https://github.com/tcbrindle/span) implementation if `std::span` of `gsl::span` unsupported
- `-D CMAKE_C_COMPILER=xxx` equal to the name of the C99 Compiler you wish to use (or the environment variable `CC`)
- `-D CMAKE_CXX_COMPILER=xxx` equal to the name of the C++17 compiler you wish to use (or the environment variable `CXX`)
- `-D CMAKE_PREFIX_PATH=xxx` to add a non-standard location for CMake to search for libraries, headers or programs
- `-D CMAKE_INSTALL_PREFIX=xxx` to install pimc to a non-standard location
- `-D STATIC=1` to enable a static build
- `-D CMAKE_BUILD_TYPE=Debug` to build deac in debug mode (deacd.e)
- `-D CMAKE_BUILD_TYPE=ZeroT` to build deac for zero temperature (deac-zT.e)
- `-D CMAKE_BUILD_TYPE=ZeroTDebug` to build deac for zero temperature in debug mode (deac-zTd.e)
- `-E env CXXFLAGS="xxx"` add additional compiler flags
- `-E env LDFLAGS="xxx"` add additional linker flags

Executables will be installed to `CMAKE_INSTALL_PREFIX` location or, if the install is skipped, they will be located in `build/deac`.
Executables produced are `deac.e`, `deacd.e`, `deac-zT.e`, and `deac-zTd.e` for `CMAKE_BUILD_TYPE=Release|Debug|ZeroT|ZeroTDebug` respectively.

A typical `cmake` line for building deac in zero temperature mode with gpu acceleration using CUDA on a Tesla V100 device with an undetected CUDA architecture and a compiler not supporting `<span>` is:

`cmake -DUSE_CUDA=1 -DGPU_BLOCK_SIZE=1024 -DCMAKE_CUDA_ARCHITECTURES=70 -DUSE_TCB_SPAN=1 -DCMAKE_BUILD_TYPE=ZeroT ../src`

If you run into problems, failures with linking etc., common errors may include
not properly setting your `LD_LIBRARY_PATH` or `LIBRARY_PATH` and not starting from a clean build
directory (issue `make clean` or `rm -rf ./*` inside the build directory).

## Usage
See `deac --help` for more details.
