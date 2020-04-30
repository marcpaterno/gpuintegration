cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cudaCuhre LANGUAGES CXX CUDA)
enable_testing()

# These flags are used by everything in the project. Put anything that affects
# the ABI here -- e.g., the standard level.
#add_compile_options(-Wall -Wextra) # -Werror excluded due to troubles in Eigen.
#add_compile_options(-Wnon-virtual-dtor -Wcast-align -Woverloaded-virtual -Wpedantic -Wdouble-promotion)
#SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )


#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-signaling-nans") # -fexcess-precision=fast")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
#add_definitions(-DOMPI_SKIP_MPICXX)

set(CMAKE_CUDA_FLAGS "-arch=sm_70")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${CMAKE_CUDA_FLAGS}")

#target_include_directories(${PROJECT_DIR})
add_executable(cudaCuhre cudaCuhre.cu)
target_link_libraries(cudaCuhre ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})