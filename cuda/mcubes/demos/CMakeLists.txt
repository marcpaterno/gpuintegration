add_executable(cuda_mcubes_c_zero_funcs c_zero_funcs.cu)
target_compile_options(cuda_mcubes_c_zero_funcs PRIVATE "-DCURAND")
set_target_properties(cuda_mcubes_c_zero_funcs PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(cuda_mcubes_gaussians gaussians.cu)
target_compile_options(cuda_mcubes_gaussians PRIVATE "-DCURAND")
set_target_properties(cuda_mcubes_gaussians PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(cuda_mcubes_discontinuous discontinuous.cu)
target_compile_options(cuda_mcubes_discontinuous PRIVATE "-DCURAND")
set_target_properties(cuda_mcubes_discontinuous PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH}) 

add_executable(cuda_mcubes_product_peaks product_peaks.cu)
target_compile_options(cuda_mcubes_product_peaks PRIVATE "-DCURAND")
set_target_properties(cuda_mcubes_product_peaks PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(cuda_mcubes_g_func g_function.cu)
target_compile_options(cuda_mcubes_g_func PRIVATE "-DCURAND")
set_target_properties(cuda_mcubes_g_func PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(cuda_mcubes_oscillatory_integrands oscillatory_integrands.cu)
target_compile_options(cuda_mcubes_oscillatory_integrands PRIVATE "-DCURAND")
set_target_properties(cuda_mcubes_oscillatory_integrands PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(cuda_mcubes_genz_integrals genz_integrals.cu)
target_compile_options(cuda_mcubes_genz_integrals PRIVATE "-DCURAND")
set_target_properties(cuda_mcubes_genz_integrals PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(cuda_mcubes_iter_10_genz_integrals iter_10_genz_integrals.cu)
target_compile_options(cuda_mcubes_iter_10_genz_integrals PRIVATE "-DCURAND")
set_target_properties(cuda_mcubes_iter_10_genz_integrals PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_SinSum6D SinSum6D.cu)
target_compile_options(mcubes_SinSum6D PRIVATE "-DCURAND")
set_target_properties(mcubes_SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz1_8D Genz1_8D.cu)
target_compile_options(mcubes_Genz1_8D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz1_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz2_6D Genz2_6D.cu)
target_compile_options(mcubes_Genz2_6D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz2_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz3_3D Genz3_3D.cu)
target_compile_options(mcubes_Genz3_3D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz3_6D Genz3_6D.cu)
target_compile_options(mcubes_Genz3_6D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz3_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz3_8D Genz3_8D.cu)
target_compile_options(mcubes_Genz3_8D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz3_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz4_5D Genz4_5D.cu)
target_compile_options(mcubes_Genz4_5D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz4_8D Genz4_8D.cu)
target_compile_options(mcubes_Genz4_8D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz4_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz5_2D Genz5_2D.cu)
target_compile_options(mcubes_Genz5_2D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz5_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz5_5D Genz5_5D.cu)
target_compile_options(mcubes_Genz5_5D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz5_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz5_8D Genz5_8D.cu)
target_compile_options(mcubes_Genz5_8D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz5_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz6_2D Genz6_2D.cu)
target_compile_options(mcubes_Genz6_2D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz6_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Genz6_6D Genz6_6D.cu)
target_compile_options(mcubes_Genz6_6D PRIVATE "-DCURAND")
set_target_properties(mcubes_Genz6_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(serial_mcubesGenz3_3D seqGenz3_3D.cu)
target_compile_options(serial_mcubesGenz3_3D PRIVATE )
set_target_properties(serial_mcubesGenz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(mcubes_Gauss9D Gauss9D.cu)
target_compile_options(mcubes_Gauss9D PRIVATE "-DCURAND")
set_target_properties(mcubes_Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(Fenc_eval_cost Fenc_eval_cost.cu)
target_compile_options(Fenc_eval_cost PRIVATE "-DCURAND")
set_target_properties(Fenc_eval_cost PROPERTIES POSITION_INDEPENDENT_CODE On CUDA_ARCHITECTURES ${TARGET_ARCH})
