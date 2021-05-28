add_executable(hello hello.cu)
set_property(TARGET hello PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(hello PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_test(hello hello)

add_executable(test_cuhre_convergence test_cuhre_convergence.cu)
set_property(TARGET test_cuhre_convergence PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(test_cuhre_convergence PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(test_cuhre_convergence util)
target_include_directories(test_cuhre_convergence PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_cuhre_convergence test_cuhre_convergence)

add_executable(BoxIntegrals BoxIntegrals.cu)
set_property(TARGET BoxIntegrals PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(BoxIntegrals PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(BoxIntegrals util)
target_include_directories(BoxIntegrals PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(BoxIntegrals BoxIntegrals)

add_executable(test_volume test_volume.cu)
target_compile_options(test_volume PRIVATE "-arch=sm_70")
set_target_properties(test_volume PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(test_volume util)
target_include_directories(test_volume PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_volume test_volume)


add_executable(RegionSampling RegionSampling.cu)
target_compile_options(RegionSampling PRIVATE "-arch=sm_70")
set_target_properties(RegionSampling PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(RegionSampling util)
target_include_directories(RegionSampling PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(RegionSampling RegionSampling)

add_executable(FourthDividedDifference FourthDividedDifference.cu)
target_compile_options(FourthDividedDifference PRIVATE "-arch=sm_70")
set_target_properties(FourthDividedDifference PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(FourthDividedDifference util)
target_include_directories(FourthDividedDifference PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(FourthDividedDifference FourthDividedDifference)

add_executable(Interpolation Interpolation.cu)
target_compile_options(Interpolation PRIVATE "-arch=sm_70")
set_target_properties(Interpolation PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Interpolation util)
target_include_directories(Interpolation PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
  ${CMAKE_SOURCE_DIR}/tests/data
)
add_test(Interpolation Interpolation)