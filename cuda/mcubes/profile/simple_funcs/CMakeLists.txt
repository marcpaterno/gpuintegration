add_executable(profile_mcubes_3D profile_mcubes_3D.cu)
target_compile_options(profile_mcubes_3D PRIVATE "-DCURAND")
set_target_properties(profile_mcubes_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(profile_mcubes_6D profile_mcubes_6D.cu)
target_compile_options(profile_mcubes_6D PRIVATE "-DCURAND")
set_target_properties(profile_mcubes_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(profile_mcubes_5D profile_mcubes_5D.cu)
target_compile_options(profile_mcubes_5D PRIVATE "-DCURAND")
set_target_properties(profile_mcubes_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(profile_mcubes_8D profile_mcubes_8D.cu)
target_compile_options(profile_mcubes_8D PRIVATE "-DCURAND")
set_target_properties(profile_mcubes_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

