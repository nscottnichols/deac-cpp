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

- `-D DEAC_MODEL=standard|hyperbolic|normalization` choose the spectral-function model. The default is `standard`, where S'(ω)=S(ω). The `hyperbolic` model uses S'(ω)=2S(ω)e^(-βω/2), and the `normalization` model uses S'(ω)=S(ω)(1 + e^(-βω)).
- `-D GPU_BACKEND=none|cuda|hip|sycl` choose the GPU backend. The default is `none`.
- `-D GPU_BLOCK_SIZE=xxx` equal to the maximum threadblock/workgroup size for GPU builds
- `-D SUB_GROUP_SIZE=xxx` equal to the subgroup size for SYCL GPU builds
- `-D MAX_GPU_STREAMS=xxx` equal to maximum number of concurrent streams or queues on GPU device
- `-D USE_BLAS=1` use backend BLAS libraries for GPU matrix operations
- `-D MKLROOT=xxx` oneMKL installation root when using `-D GPU_BACKEND=sycl -D USE_BLAS=1` and `MKLROOT` is not already set in the environment
- `-D USE_BOSONIC_DETAILED_BALANCE_CONDITION_DSF=1` build the bosonic detailed-balance DSF kernel variant
- `-D SINGLE_PARTICLE_FERMIONIC_SPECTRAL_FUNCTION=1` enable single-particle fermionic spectral-function behavior
- `-D USE_ALLOW_NEGATIVE_SPECTRAL_WEIGHT=1` allow negative spectral weights
- `-D USE_SYCL_MATMUL_WA=1` enable the SYCL matrix-multiplication workaround
- `-D CMAKE_CUDA_ARCHITECTURES=xxx` equal to CUDA device architecture if not properly detected by CMake
- `-D SYCL_FLAGS="xxx"` target architecture and device flags for SYCL builds
- `-D CMAKE_C_COMPILER=xxx` equal to the name of the C99 Compiler you wish to use (or the environment variable `CC`)
- `-D CMAKE_CXX_COMPILER=xxx` equal to the name of the C++20 compiler you wish to use (or the environment variable `CXX`)
- `-D CMAKE_PREFIX_PATH=xxx` to add a non-standard location for CMake to search for libraries, headers or programs
- `-D CMAKE_INSTALL_PREFIX=xxx` to install deac to a non-standard location
- `-D STATIC=1` to enable a static build
- `-D CMAKE_BUILD_TYPE=Debug` to build deac in debug mode (deacd.e)
- `-D CMAKE_BUILD_TYPE=ZeroT` to build deac for zero temperature (deac-zT.e)
- `-D CMAKE_BUILD_TYPE=ZeroTDebug` to build deac for zero temperature in debug mode (deac-zTd.e)
- `-E env CXXFLAGS="xxx"` add additional compiler flags
- `-E env LDFLAGS="xxx"` add additional linker flags

Executables will be installed to `CMAKE_INSTALL_PREFIX` location or, if the install is skipped, they will be located in `build/deac`.
Executables produced are `deac.e`, `deacd.e`, `deac-zT.e`, and `deac-zTd.e` for `CMAKE_BUILD_TYPE=Release|Debug|ZeroT|ZeroTDebug` respectively.

A typical `cmake` line for building deac in zero temperature mode with GPU acceleration using CUDA on a Tesla V100 device with an undetected CUDA architecture is:

`cmake -DGPU_BACKEND=cuda -DGPU_BLOCK_SIZE=1024 -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=ZeroT ../src`

If you run into problems, failures with linking etc., common errors may include
not properly setting your `LD_LIBRARY_PATH` or `LIBRARY_PATH` and not starting from a clean build
directory (issue `make clean` or `rm -rf ./*` inside the build directory).

## Usage and analysis

### Step 1: set up input data

1. get data into numpy arrays (imaginary_time_array, intermediate_scattering_function_array, intermediate_scattering_function_error_array)
2. save data as npz file
        `np.savez(filename, tau = imaginary_time_array, isf = intermediate_scattering_function_array, error = intermediate_scattering_function_error_array)`
3. run converter script in `tools` directory to convert `.npz` file to binary format used by deac
        `/path/to/DEAC_TOOLS_DIR/convert_isf_data.py /path/to/isf_npz_file /path/to/isf_bin_file`

### Step 2: run DEAC

A single reconstruction of the dynamic structure factor can be generated using the input data from step 1. For example:
```bash
deac.e --number_of_generations 1600000 --temperature 1.35 --population_size 8 --genome_size 4096 --normalize --omega_max 512.0 --save_directory deac_results --seed 1 --stop_minimum_fitness 1.0 isf.bin
```
This command will save the generated spectrum in the `deac_results` directory after `1600000` steps or if the desired fitness cutoff of `1.0` is reached. Please see `deac.e --help` for more command line arguments and further description.
	
The general recommendation is to generate several spectra using different seeds (`--seed`) and average the final results. A sample bash script to generate a command file with multiple seeds can be found in the tools directory `tools/generate_commands.sh`.
	
### Step 3: analyze DEAC results

The data generated by DEAC is in binary format and can be loaded into Python using `numpy`.
```python
frequency_filename = "/path/to/deac_frequency_file.bin"
dynamic_structure_factor_filename = "/path/to/deac_dsf_file.bin"
frequency = np.fromfile(frequency_filename, dtype=np.double)
dsf = np.fromfile(dynamic_structure_factor_filename, dtype=np.double)
```
An example script to plot the average for multiple spectra using many seeds (with results saved in the same directory) can be found in the tools directory (`tools/plot_simple_average_deac_results.py`).

See `deac --help` for more details.
