cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cudaCuhre LANGUAGES CUDA CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(QUAD_WARNINGS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cudaCuhre cudaCuhre.cu)
set_property(TARGET cudaCuhre PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(cudaCuhre PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cudaCuhre util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

add_subdirectory(quad)
add_subdirectory(tests)
