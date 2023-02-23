cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpuintegration VERSION 0.1.0 LANGUAGES CXX CUDA C)
enable_testing()

# These flags are used by everything in the project. Put anything that affects
# the ABI here -- e.g., the standard level.
SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Turn on compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
SET(CMAKE_CUDA_FLAGS_DEBUG "-g -lineinfo")

# TODO: Modernize CMake usage; we should not use a global "include_directories"
if (GPUINTEGRATION_BUILD_ONEAPI)
  include_directories(AFTER cubacpp externals ${CMAKE_SOURCE_DIR}  ${DPCT_DIR} ${DPL_DIR} ${SYCL_DIR} ${ONEMKL_DIR})
else()
  include_directories(AFTER cubacpp externals ${CMAKE_SOURCE_DIR})
endif()

if ((NOT GPUINTEGRATION_BUILD_CUDA) AND
    (NOT GPUINTEGRATION_BUILD_KOKKOS) AND
    (NOT GPUINTEGRATION_BUILD_ONEAPI))
    message(FATAL_ERROR "No back-end selected")
endif()

if (GPUINTEGRATION_BUILD_CUDA)
  message(STATUS "Building the CUDA back-end")
  if (NOT TARGET_ARCH)
    message(FATAL_ERROR "You must specify TARGET_ARCH for CUDA builds")
  endif()
  add_subdirectory(cuda)
endif()

if (GPUINTEGRATION_BUILD_KOKKOS)
  find_package(Kokkos REQUIRED)
  find_package(KokkosKernels REQUIRED)
  message(STATUS "Building the Kokkos back-end")
  add_subdirectory(kokkos)
endif()

if (GPUINTEGRATION_BUILD_ONEAPI)
  message(STATUS "Building the oneAPI back-end")
 #if(CUDA_BACKEND)
	#set(CMAKE_CXX_COMPILER clang++)
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70")
 #endif()
  add_subdirectory(oneAPI)
  add_subdirectory(dpct-exp)
endif()

add_subdirectory(test)
