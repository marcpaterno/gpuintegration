add_executable(SinSum6D SinSum6D.cu)
target_compile_options(SinSum6D PRIVATE "-arch=sm_70" "-O2" "-DCURAND")
set_target_properties(SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(Gauss9D Gauss9D.cu)
target_compile_options(Gauss9D PRIVATE "-arch=sm_70" "-O2" "-DCURAND")
set_target_properties(Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE ON)