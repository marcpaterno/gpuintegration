if(CUDA_BACKEND)
	set(CMAKE_CXX_COMPILER clang++)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda -lonemkl -Xsycl-target-backend --cuda-gpu-arch=${ONEAPI_TARGET_ARCH}")
endif()

add_subdirectory(demos)
