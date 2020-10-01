cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpuintegration VERSION 0.0.1 LANGUAGES CXX CUDA)
enable_testing()

# These flags are used by everything in the project. Put anything that affects
# the ABI here -- e.g., the standard level.
SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(AFTER cubacpp externals y3_cluster_cpp cudaCuhre "/opt/openmpi-wahab/include" "/opt/gsl")

set(CMAKE_SHARED_MODULE_PREFIX "")    # We don't need the leading lib

find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(cubacpp)
add_subdirectory(cudaCuhre)
add_subdirectory(y3_cluster_cpp)

