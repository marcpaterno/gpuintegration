add_subdirectory(simple_funcs)
add_subdirectory(sum_of_args_integrands)

add_executable(oneapi_profile_SinSum oneapi_profile_SinSum.cpp)
target_compile_options(oneapi_profile_SinSum PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(profile_Genz3_3D oneapi_profile_Genz3_3D.cpp)
target_compile_options(profile_Genz3_3D PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(profile_Genz3_8D oneapi_profile_Genz3_8D.cpp)
target_compile_options(profile_Genz3_8D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_Genz2_6D oneapi_profile_Genz2_6D.cpp)
target_compile_options(profile_Genz2_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_Genz4_5D oneapi_profile_Genz4_5D.cpp)
target_compile_options(profile_Genz4_5D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_Genz4_6D oneapi_profile_Genz4_6D.cpp)
target_compile_options(profile_Genz4_6D PRIVATE "-mllvm" "-inline-threshold=10000" "-lonemkl" "-ltbb")

add_executable(profile_Genz5_8D oneapi_profile_Genz5_8D.cpp)
target_compile_options(profile_Genz5_8D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_Genz6_6D oneapi_profile_Genz6_6D.cpp)
target_compile_options(profile_Genz6_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(execute_8D_benchmark_integrands_on_device execute_8D_benchmark_integrands_on_device.cpp)
target_compile_options(execute_8D_benchmark_integrands_on_device PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(execute_7D_benchmark_integrands_on_device execute_7D_benchmark_integrands_on_device.cpp)
target_compile_options(execute_7D_benchmark_integrands_on_device PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(execute_6D_benchmark_integrands_on_device execute_6D_benchmark_integrands_on_device.cpp)
target_compile_options(execute_6D_benchmark_integrands_on_device PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(execute_5D_benchmark_integrands_on_device execute_5D_benchmark_integrands_on_device.cpp)
target_compile_options(execute_5D_benchmark_integrands_on_device PRIVATE "-mllvm" "-inline-threshold=10000")


add_executable(execute_math_functions_on_device execute_math_functions_on_device.cpp)
target_compile_options(execute_math_functions_on_device PRIVATE "-mllvm" "-inline-threshold=10000")
#target_compile_options(execute_math_functions_on_device PRIVATE "-mllvm" "-inline-threshold=10000" "-O3" "-ffast-math")

add_executable(profile_integrands profile_integrands.cpp)
target_compile_options(profile_integrands PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(profile_mcubes_integrands profile_mcubes_integrands.cpp)
target_compile_options(profile_mcubes_integrands PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(profile_pagani_integrands profile_pagani_integrands.cpp)
target_compile_options(profile_pagani_integrands PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(atomic_addition atomic_addition.cpp)
target_compile_options(atomic_addition PRIVATE "-mllvm" "-inline-threshold=10000")
