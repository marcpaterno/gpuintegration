add_executable(oneapi_Interpolation1D Interpolation1D.dp.cpp)
#target_link_libraries(Interpolation1D PRIVATE MKL::MKL_DPCPP)
target_compile_options(oneapi_Interpolation1D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_Interpolation1D oneapi_Interpolation1D)

add_executable(oneapi_Interpolation2D Interpolation2D.dp.cpp)
#target_link_libraries(Interpolation2D PRIVATE MKL::MKL_DPCPP)
target_compile_options(oneapi_Interpolation2D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_Interpolation2D oneapi_Interpolation2D)

add_executable(oneapi_NanIntegral NanIntegral.dp.cpp)
#target_link_libraries(NanIntegral PRIVATE MKL::MKL_DPCPP)
target_compile_options(oneapi_NanIntegral PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_NanIntegral oneapi_NanIntegral)

add_executable(oneapi_uniform_functions uniform_functions.cpp)
#target_link_libraries(uniform_functions PRIVATE MKL::MKL_DPCPP)
target_compile_options(oneapi_uniform_functions PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_uniform_functions oneapi_uniform_functions)

add_executable(oneapi_Uniform_sub_division Uniform_sub_division.cpp)
#target_link_libraries(Uniform_sub_division PRIVATE MKL::MKL_DPCPP)
target_compile_options(oneapi_Uniform_sub_division PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_Uniform_sub_division oneapi_Uniform_sub_division)

add_executable(oneapi_RegionSampling RegionSampling.dp.cpp)
#target_link_libraries(RegionSampling PRIVATE MKL::MKL_DPCPP)
target_compile_options(oneapi_RegionSampling PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_RegionSampling oneapi_RegionSampling)


add_executable(oneapi_Reduction Reduction.cpp)
#target_link_libraries(Reduction PRIVATE MKL::MKL_DPCPP)
target_compile_options(oneapi_Reduction PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_Reduction oneapi_Reduction)

add_executable(oneapi_prefix_sum prefix_sum.cpp)
#target_link_libraries(prefix_sum PRIVATE MKL::MKL_DPCPP)
target_compile_options(oneapi_prefix_sum PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_prefix_sum oneapi_prefix_sum)

add_executable(oneapi_min_max min_max.cpp)
#target_link_libraries(min_max PRIVATE MKL::MKL_DPCPP)
target_compile_options(oneapi_min_max PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_min_max oneapi_min_max)

add_executable(oneapi_StatefulIntegrand StatefulIntegrand.dp.cpp)
target_compile_options(oneapi_StatefulIntegrand PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
add_test(oneapi_StatefulIntegrand oneapi_StatefulIntegrand)