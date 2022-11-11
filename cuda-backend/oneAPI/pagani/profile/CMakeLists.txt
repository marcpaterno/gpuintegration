add_executable(profile_Genz3_3D oneapi_profile_Genz3_3D.cpp)
target_compile_options(profile_Genz3_3D PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(profile_Genz3_8D oneapi_profile_Genz3_8D.cpp)
target_compile_options(profile_Genz3_8D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_Genz2_6D oneapi_profile_Genz2_6D.cpp)
target_compile_options(profile_Genz2_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_Genz4_5D oneapi_profile_Genz4_5D.cpp)
target_compile_options(profile_Genz4_5D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_Genz4_6D oneapi_profile_Genz4_6D.cpp)
target_compile_options(profile_Genz4_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_Genz5_8D oneapi_profile_Genz5_8D.cpp)
target_compile_options(profile_Genz5_8D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_Genz6_6D oneapi_profile_Genz6_6D.cpp)
target_compile_options(profile_Genz6_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(oneapi_profile_pagani_pow_of_sum oneapi_profile_pow_of_sum.cpp)
target_compile_options(oneapi_profile_pagani_pow_of_sum PRIVATE "-mllvm" "-inline-threshold=10000")
