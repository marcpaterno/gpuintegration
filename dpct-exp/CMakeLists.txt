if (NOT DEFINED ONEAPI_TARGET_ARCH AND CUDA_BACKEND)
	MESSAGE(FATAL_ERROR "You must specify -DONEAPI_TARGET_ARCH=<arch> to generate code for the right GPU, e.g. '=sm_70' for V100")
endif()



add_subdirectory(cuda)
