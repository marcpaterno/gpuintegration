add_executable(SinSum6D SinSum6D.cu)
target_compile_options(SinSum6D PRIVATE "-O2" "-DCURAND")
set_target_properties(SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz1_8D Genz1_8D.cu)
target_compile_options(Genz1_8D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz1_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz2_6D Genz2_6D.cu)
target_compile_options(Genz2_6D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz2_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz3_3D Genz3_3D.cu)
target_compile_options(Genz3_3D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz3_6D Genz3_6D.cu)
target_compile_options(Genz3_6D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz3_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz3_8D Genz3_8D.cu)
target_compile_options(Genz3_8D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz3_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz4_5D Genz4_5D.cu)
target_compile_options(Genz4_5D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz4_8D Genz4_8D.cu)
target_compile_options(Genz4_8D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz4_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz5_2D Genz5_2D.cu)
target_compile_options(Genz5_2D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz5_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz5_5D Genz5_5D.cu)
target_compile_options(Genz5_5D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz5_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz5_8D Genz5_8D.cu)
target_compile_options(Genz5_8D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz5_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz6_2D Genz6_2D.cu)
target_compile_options(Genz6_2D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz6_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(Genz6_6D Genz6_6D.cu)
target_compile_options(Genz6_6D PRIVATE "-O2" "-DCURAND")
set_target_properties(Genz6_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(smcubesGenz3_3D seqGenz3_3D.cu)
target_compile_options(smcubesGenz3_3D PRIVATE "-O2")
set_target_properties(smcubesGenz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
add_executable(Gauss9D Gauss9D.cu)
target_compile_options(Gauss9D PRIVATE "-O2" "-DCURAND")
set_target_properties(Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
