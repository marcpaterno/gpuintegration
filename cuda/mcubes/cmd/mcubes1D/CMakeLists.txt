cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mcubes1D VERSION 0.0.1 LANGUAGES CXX CUDA C)
enable_testing()

# These flags are used by everything in the project. Put anything that affects
# the ABI here -- e.g., the standard level.
SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Turn on compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

include_directories(AFTER ${CMAKE_SOURCE_DIR})

set(CMAKE_SHARED_MODULE_PREFIX "")    # We don't need the leading lib

add_executable(vegas1D vegas_mcubes_1D.cu)
target_compile_options(vegas1D PRIVATE "-arch=sm_70" "-O2" "-DCURAND")
set_target_properties(vegas1D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vegas1D PRIVATE ${CMAKE_SOURCE_DIR})


add_executable(vegas1Dcustom vegas_mcubes_1D.cu)
target_compile_options(vegas1Dcustom PRIVATE "-arch=sm_70" "-O2" "-DCUSTOM")
set_target_properties(vegas1Dcustom PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vegas1Dcustom PRIVATE ${CMAKE_SOURCE_DIR})