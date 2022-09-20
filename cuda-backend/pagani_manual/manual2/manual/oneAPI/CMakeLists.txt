#cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
#project(gpuintegration VERSION 0.0.1 LANGUAGES CXX C)
#enable_testing()

# These flags are used by everything in the project. Put anything that affects
# the ABI here -- e.g., the standard level.
#SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )

#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Turn on compiler warnings
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

#find_package(IntelDPCPP REQUIRED)

include_directories(AFTER ${CMAKE_SOURCE_DIR} ${PAGANI_DIR})

set(CMAKE_SHARED_MODULE_PREFIX "")    # We don't need the leading lib

add_subdirectory(demos)
add_subdirectory(tests)
#https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compiler-setup/using-the-command-line/using-cmake-with-data-parallel-c-dpc.html?wapkw=cmake
# says that we need to supply like this:cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx
# or CC=icx CXX=icx cmake ...
