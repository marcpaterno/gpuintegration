cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(gpuintegration VERSION 0.1.0 LANGUAGES CXX)
enable_testing()

# These flags are used by everything in the project. Put anything that affects
# the ABI here -- e.g., the standard level.
#SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )
SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} " )

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Turn on compiler warnings
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

include_directories(AFTER ${EXTERNALS_DIR} ${TOP_DIR} "/opt/intel/compiler/2022.0.2/linux/include/sycl" "/home/u133027/numint/dpct/oneAPI/Catch2/src/catch2")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

#find_package(IntelDPCPP REQUIRED)
#find_package(TBB REQUIRED)
#find_package(MKL REQUIRED)

add_subdirectory(oneAPI)
