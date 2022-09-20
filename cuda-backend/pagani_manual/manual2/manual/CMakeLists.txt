cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(gpuintegration VERSION 0.1.0 LANGUAGES CXX)
enable_testing()

# These flags are used by everything in the project. Put anything that affects
# the ABI here -- e.g., the standard level.
SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Turn on compiler warnings
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(TBB REQUIRED)
find_package(MKL REQUIRED)

# TODO: Modernize CMake usage; we should not use a global "include_directories"
include_directories(AFTER ${PAGANI_DIR})
#target_link_libraries(foo TBB::tbb)
add_subdirectory(oneAPI)




