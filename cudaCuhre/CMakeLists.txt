cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cudaCuhre LANGUAGES CUDA CXX)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(QUAD_WARNINGS)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cudaCuhre cudaCuhre.cu)
target_compile_features(cudaCuhre PUBLIC cxx_std_14)
target_compile_options(cudaCuhre PRIVATE "-arch=sm_70")
set_target_properties(cudaCuhre PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cudaCuhre util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

add_subdirectory(quad)
add_subdirectory(tests)
