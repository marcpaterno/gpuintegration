find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(hello hello.cu)
set_property(TARGET hello PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(hello PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_test(hello hello)

add_library(do_integration_from_c do_integration_from_c.cu)
set_property(TARGET do_integration_from_c PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(do_integration_from_c PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(do_integration_from_c util ${NVTX_LIBRARY})
target_include_directories(do_integration_from_c PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(test_integration_from_c test_integration_from_c.c)
target_link_libraries(test_integration_from_c do_integration_from_c)
add_test(test_integration_from_c test_integration_from_c)

add_executable(StatefullIntegrand StatefulIntegrand.cu)
set_property(TARGET StatefullIntegrand PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(StatefullIntegrand PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(StatefullIntegrand quadutil ${NVTX_LIBRARY})
target_include_directories(StatefullIntegrand PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(StatefullIntegrand StatefullIntegrand)


add_executable(test_cuhre_convergence test_cuhre_convergence.cu)
set_property(TARGET test_cuhre_convergence PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(test_cuhre_convergence PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(test_cuhre_convergence util ${NVTX_LIBRARY})
target_include_directories(test_cuhre_convergence PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_cuhre_convergence test_cuhre_convergence)

add_executable(BoxIntegrals BoxIntegrals.cu)
set_property(TARGET BoxIntegrals PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(BoxIntegrals PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(BoxIntegrals util ${NVTX_LIBRARY})
target_include_directories(BoxIntegrals PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(BoxIntegrals BoxIntegrals)

add_executable(test_volume test_volume.cu)
target_compile_options(test_volume PRIVATE "-arch=sm_70")
set_target_properties(test_volume PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(test_volume util ${NVTX_LIBRARY})
target_include_directories(test_volume PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_volume test_volume)


add_executable(RegionSampling RegionSampling.cu)
target_compile_options(RegionSampling PRIVATE "-arch=sm_70")
set_target_properties(RegionSampling PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(RegionSampling util ${NVTX_LIBRARY})
target_include_directories(RegionSampling PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(RegionSampling RegionSampling)

add_executable(FourthDividedDifference FourthDividedDifference.cu)
target_compile_options(FourthDividedDifference PRIVATE "-arch=sm_70")
set_target_properties(FourthDividedDifference PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(FourthDividedDifference util ${NVTX_LIBRARY})
target_include_directories(FourthDividedDifference PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(FourthDividedDifference FourthDividedDifference)

add_executable(Interpolation Interpolation.cu)
target_compile_options(Interpolation PRIVATE "-arch=sm_70")
set_target_properties(Interpolation PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Interpolation util ${NVTX_LIBRARY})
target_include_directories(Interpolation PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
  ${CMAKE_SOURCE_DIR}/tests/data
)
add_test(Interpolation Interpolation)


add_executable(Interpolation1D Interpolation1D.cu)
set_property(TARGET Interpolation1D PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(Interpolation1D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Interpolation1D quadutil ${NVTX_LIBRARY})
target_include_directories(Interpolation1D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Interpolation1D Interpolation1D)

add_executable(Interpolation2D Interpolation2D.cu)
set_property(TARGET Interpolation2D PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(Interpolation2D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Interpolation2D quadutil ${NVTX_LIBRARY})
target_include_directories(Interpolation2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Interpolation2D Interpolation2D)