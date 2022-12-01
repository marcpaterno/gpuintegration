add_executable(profile_3D oneapi_profile_3D.cpp)
target_compile_options(profile_3D PRIVATE "-mllvm" "-inline-threshold=10000")

add_executable(profile_5D oneapi_profile_5D.cpp)
target_compile_options(profile_5D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_6D oneapi_profile_6D.cpp)
target_compile_options(profile_6D PRIVATE "-mllvm" "-inline-threshold=10000" )

add_executable(profile_8D oneapi_profile_8D.cpp)
target_compile_options(profile_8D PRIVATE "-mllvm" "-inline-threshold=10000" )


