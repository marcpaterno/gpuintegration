add_executable(SinSum6D SinSum6D.cu)
target_compile_options(SinSum6D PRIVATE "-O2" "-DCURAND")
set_target_properties(SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Gauss9D Gauss9D.cu)
target_compile_options(Gauss9D PRIVATE "-O2" "-DCURAND")
set_target_properties(Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
