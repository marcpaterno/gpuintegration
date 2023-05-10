
if (GPUINTEGRATION_BUILD_CUDA)
  #set(CMAKE_CXX_COMPILER g++)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  add_subdirectory(cuda)
endif()

if (GPUINTEGRATION_BUILD_KOKKOS)
  #set(CMAKE_CXX_COMPILER g++)
  set(CMAKE_CXX_FLAGS "-Wall -Wextra")
  add_subdirectory(kokkos)
endif()


if (GPUINTEGRATION_BUILD_ONEAPI)
  if(CUDA_BACKEND)
	set(CMAKE_CXX_COMPILER clang++)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -lonemkl -Xsycl-target-backend --cuda-gpu-arch=${ONEAPI_TARGET_ARCH}")
 endif()
  add_subdirectory(oneAPI)
endif()