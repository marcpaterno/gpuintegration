add_subdirectory(simple_funcs)
add_subdirectory(sum_of_args_integrands)

add_executable(profile_oneAPI_mcubes_Genz2_6D Genz2_6D.dp.cpp)
target_compile_options(profile_oneAPI_mcubes_Genz2_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_oneAPI_mcubes_Genz2_8D Genz2_8D.dp.cpp)
target_compile_options(profile_oneAPI_mcubes_Genz2_8D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_oneAPI_mcubes_Genz3_3D Genz3_3D.dp.cpp)
target_compile_options(profile_oneAPI_mcubes_Genz3_3D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_oneAPI_mcubes_Genz3_8D Genz3_8D.dp.cpp)
target_compile_options(profile_oneAPI_mcubes_Genz3_8D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_oneAPI_mcubes_Genz4_5D Genz4_5D.dp.cpp)
target_compile_options(profile_oneAPI_mcubes_Genz4_5D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_oneAPI_mcubes_Genz5_8D Genz5_8D.dp.cpp)
target_compile_options(profile_oneAPI_mcubes_Genz5_8D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_oneAPI_mcubes_Genz6_6D Genz6_6D.dp.cpp)
target_compile_options(profile_oneAPI_mcubes_Genz6_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(oneapi_profile_mcubes_pow_of_sum oneapi_profile_pow_of_sum.cpp)
target_compile_options(oneapi_profile_mcubes_pow_of_sum PRIVATE "-mllvm" "-inline-threshold=10000")
