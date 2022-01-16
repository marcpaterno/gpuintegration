cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(mcubes VERSION 0.0.1 LANGUAGES CXX CUDA C)
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

if (NOT DEFINED PAGANI_TARGET_ARCH)
	MESSAGE(FATAL_ERROR "You must specify -DPAGANI_TARGET_ARCH=<arch> to generate code for the right GPU, e.g. '=70-real' for V100")
endif()

include_directories(AFTER ${PAGANI_DIR} ${CMAKE_SOURCE_DIR} ${VEGAS_DIR} externals)

set(CMAKE_SHARED_MODULE_PREFIX "")    # We don't need the leading lib

add_subdirectory(demos)
add_subdirectory(tests)
