add_executable(Interpolation1D Interpolation1D.dp.cpp)
target_link_libraries(Interpolation1D PRIVATE MKL::MKL_DPCPP)
add_test(Interpolation1D Interpolation1D)

add_executable(Interpolation2D Interpolation2D.dp.cpp)
target_link_libraries(Interpolation2D PRIVATE MKL::MKL_DPCPP)
add_test(Interpolation2D Interpolation2D)

add_executable(NanIntegral NanIntegral.dp.cpp)
target_link_libraries(NanIntegral PRIVATE MKL::MKL_DPCPP)
add_test(NanIntegral NanIntegral)

