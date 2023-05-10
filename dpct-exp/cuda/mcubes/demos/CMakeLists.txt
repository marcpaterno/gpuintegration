add_executable(dpct_repeat_mcubes_Genz2_6D Genz2_6D.dp.cpp)
target_compile_options(dpct_repeat_mcubes_Genz2_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_dpct_repeat_mcubes_kernel_Genz2_6D profile_Genz2_6D.dp.cpp)
target_compile_options(profile_dpct_repeat_mcubes_kernel_Genz2_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_dpct_repeat_mcubes_kernel_Genz2_8D profile_Genz2_8D.dp.cpp)
target_compile_options(profile_dpct_repeat_mcubes_kernel_Genz2_8D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_dpct_repeat_mcubes_kernel_Genz6_6D Genz6_6D.dp.cpp)
target_compile_options(profile_dpct_repeat_mcubes_kernel_Genz6_6D PRIVATE "-mllvm" "-inline-threshold=10000" )