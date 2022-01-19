add_executable(mcubes_SinSum6D SinSum6D.cu)
target_compile_options(mcubes_SinSum6D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz1_8D Genz1_8D.cu)
target_compile_options(mcubes_Genz1_8D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz1_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz2_6D Genz2_6D.cu)
target_compile_options(mcubes_Genz2_6D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz2_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz3_3D Genz3_3D.cu)
target_compile_options(mcubes_Genz3_3D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz3_6D Genz3_6D.cu)
target_compile_options(mcubes_Genz3_6D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz3_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz3_8D Genz3_8D.cu)
target_compile_options(mcubes_Genz3_8D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz3_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz4_5D Genz4_5D.cu)
target_compile_options(mcubes_Genz4_5D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz4_8D Genz4_8D.cu)
target_compile_options(mcubes_Genz4_8D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz4_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz5_2D Genz5_2D.cu)
target_compile_options(mcubes_Genz5_2D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz5_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz5_5D Genz5_5D.cu)
target_compile_options(mcubes_Genz5_5D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz5_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz5_8D Genz5_8D.cu)
target_compile_options(mcubes_Genz5_8D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz5_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz6_2D Genz6_2D.cu)
target_compile_options(mcubes_Genz6_2D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz6_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Genz6_6D Genz6_6D.cu)
target_compile_options(mcubes_Genz6_6D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Genz6_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(serial_mcubesGenz3_3D seqGenz3_3D.cu)
target_compile_options(serial_mcubesGenz3_3D PRIVATE "-O2")
set_target_properties(serial_mcubesGenz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(mcubes_Gauss9D Gauss9D.cu)
target_compile_options(mcubes_Gauss9D PRIVATE "-O2" "-DCURAND")
set_target_properties(mcubes_Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})