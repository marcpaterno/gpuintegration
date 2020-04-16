cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(y3_cluster_cpp VERSION 0.2.0 LANGUAGES CXX)
enable_testing()

# These flags are used by everything in the project. Put anything that affects
# the ABI here -- e.g., the standard level.
add_compile_options(-W -Wall -Wextra) # -Werror excluded due to troubles in Eigen.
add_compile_options(-Wnon-virtual-dtor -Wcast-align -Woverloaded-virtual -Wpedantic -Wdouble-promotion)
SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mcpu=native -mtune=native" )

if (CMAKE_CXX_COMPILER STREQUAL "gcc")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-signaling-nans -fexcess-precision=fast")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_SHARED_MODULE_SUFFIX ".so") # We want this on all operating systems
set(CMAKE_SHARED_MODULE_PREFIX "")    # We don't need the leading lib
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

find_package(cuba REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(models)
add_subdirectory(modules)
add_subdirectory(tests)
add_subdirectory(utils)
