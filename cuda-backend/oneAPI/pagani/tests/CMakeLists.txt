add_executable(Interpolation1D Interpolation1D.dp.cpp)
target_link_libraries(Interpolation1D PRIVATE MKL::MKL_DPCPP)
add_test(Interpolation1D Interpolation1D)

add_executable(Interpolation2D Interpolation2D.dp.cpp)
target_link_libraries(Interpolation2D PRIVATE MKL::MKL_DPCPP)
add_test(Interpolation2D Interpolation2D)

add_executable(NanIntegral NanIntegral.dp.cpp)
target_link_libraries(NanIntegral PRIVATE MKL::MKL_DPCPP)
add_test(NanIntegral NanIntegral)

add_executable(uniform_functions uniform_functions.cpp)
target_link_libraries(uniform_functions PRIVATE MKL::MKL_DPCPP)
add_test(uniform_functions uniform_functions)

add_executable(Uniform_sub_division Uniform_sub_division.cpp)
target_link_libraries(Uniform_sub_division PRIVATE MKL::MKL_DPCPP)
add_test(Uniform_sub_division Uniform_sub_division)

add_executable(RegionSampling RegionSampling.dp.cpp)
target_link_libraries(RegionSampling PRIVATE MKL::MKL_DPCPP)
add_test(RegionSampling RegionSampling)


add_executable(Reduction Reduction.cpp)
target_link_libraries(Reduction PRIVATE MKL::MKL_DPCPP)
add_test(Reduction Reduction)

add_executable(prefix_sum prefix_sum.cpp)
target_link_libraries(prefix_sum PRIVATE MKL::MKL_DPCPP)
add_test(prefix_sum prefix_sum)

add_executable(min_max min_max.cpp)
target_link_libraries(min_max PRIVATE MKL::MKL_DPCPP)
add_test(min_max min_max)