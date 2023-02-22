if (NOT DEFINED ONEAPI_TARGET_ARCH AND CUDA_BACKEND)
	MESSAGE(FATAL_ERROR "You must specify -DONEAPI_TARGET_ARCH=<arch> to generate code for the right GPU, e.g. '=sm_70' for V100")
endif()

if(CUDA_BACKEND)
	set(CMAKE_CXX_COMPILER clang++)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mllvm -inline-threshold=10000 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -lonemkl -Xsycl-target-backend --cuda-gpu-arch=${ONEAPI_TARGET_ARCH}")
endif()

add_subdirectory(cuda)
