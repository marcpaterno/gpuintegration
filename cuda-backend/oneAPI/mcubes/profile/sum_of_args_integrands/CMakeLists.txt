add_executable(oneapi_profile_mcubes_kernel_f_3D oneapi_profile_mcubes_kernel_f_3D.cpp)
target_compile_options(oneapi_profile_mcubes_kernel_f_3D PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(oneapi_profile_mcubes_kernel_f_5D oneapi_profile_mcubes_kernel_f_5D.cpp)
target_compile_options(oneapi_profile_mcubes_kernel_f_5D PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(oneapi_profile_mcubes_kernel_f_6D oneapi_profile_mcubes_kernel_f_6D.cpp)
target_compile_options(oneapi_profile_mcubes_kernel_f_3D PRIVATE "-mllvm" "-inline-threshold=10000")


add_executable(oneapi_profile_mcubes_kernel_f_8D oneapi_profile_mcubes_kernel_f_8D.cpp)
target_compile_options(oneapi_profile_mcubes_kernel_f_8D PRIVATE "-mllvm" "-inline-threshold=10000")
