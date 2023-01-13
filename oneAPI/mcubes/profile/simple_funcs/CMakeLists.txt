add_executable(oneapi_profile_oneAPI_mcubes_3D profile_3D.dp.cpp)
target_compile_options(oneapi_profile_oneAPI_mcubes_3D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(oneapi_profile_oneAPI_mcubes_5D profile_5D.dp.cpp)
target_compile_options(oneapi_profile_oneAPI_mcubes_5D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(oneapi_profile_oneAPI_mcubes_6D profile_6D.dp.cpp)
target_compile_options(oneapi_profile_oneAPI_mcubes_6D PRIVATE "-mllvm" "-inline-threshold=10000" )


add_executable(oneapi_profile_oneAPI_mcubes_8D profile_8D.dp.cpp)
target_compile_options(oneapi_profile_oneAPI_mcubes_8D PRIVATE "-mllvm" "-inline-threshold=10000" )

