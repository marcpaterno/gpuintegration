add_executable(Interpolation1D Interpolation1D.dp.cpp)
#target_link_libraries(Interpolation1D PRIVATE MKL::MKL_DPCPP)
target_compile_options(Interpolation1D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(Interpolation1D Interpolation1D)

add_executable(Interpolation2D Interpolation2D.dp.cpp)
#target_link_libraries(Interpolation2D PRIVATE MKL::MKL_DPCPP)
target_compile_options(Interpolation2D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(Interpolation2D Interpolation2D)

add_executable(NanIntegral NanIntegral.dp.cpp)
#target_link_libraries(NanIntegral PRIVATE MKL::MKL_DPCPP)
target_compile_options(NanIntegral PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(NanIntegral NanIntegral)

add_executable(uniform_functions uniform_functions.cpp)
#target_link_libraries(uniform_functions PRIVATE MKL::MKL_DPCPP)
target_compile_options(uniform_functions PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(uniform_functions uniform_functions)

add_executable(Uniform_sub_division Uniform_sub_division.cpp)
#target_link_libraries(Uniform_sub_division PRIVATE MKL::MKL_DPCPP)
target_compile_options(Uniform_sub_division PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(Uniform_sub_division Uniform_sub_division)

add_executable(RegionSampling RegionSampling.dp.cpp)
#target_link_libraries(RegionSampling PRIVATE MKL::MKL_DPCPP)
target_compile_options(RegionSampling PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(RegionSampling RegionSampling)


add_executable(Reduction Reduction.cpp)
#target_link_libraries(Reduction PRIVATE MKL::MKL_DPCPP)
target_compile_options(Reduction PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(Reduction Reduction)

add_executable(prefix_sum prefix_sum.cpp)
#target_link_libraries(prefix_sum PRIVATE MKL::MKL_DPCPP)
target_compile_options(prefix_sum PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(prefix_sum prefix_sum)

add_executable(min_max min_max.cpp)
#target_link_libraries(min_max PRIVATE MKL::MKL_DPCPP)
target_compile_options(min_max PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(min_max min_max)