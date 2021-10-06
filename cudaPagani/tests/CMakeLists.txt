find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(CuhrePointsEvaluation CuhrePointsEvaluation.cu)
set_target_properties(CuhrePointsEvaluation PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_include_directories(CuhrePointsEvaluation PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(CuhrePointsEvaluation CuhrePointsEvaluation)

add_executable(cudaUnifiedArray cudaUnifiedArray.cu)
target_link_libraries(cudaUnifiedArray cpu_model gpu_model)
set_target_properties(cudaUnifiedArray PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_include_directories(cudaUnifiedArray PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(cudaUnifiedArray cudaUnifiedArray)



add_library(cpu_model model.cc)
target_include_directories(cpu_model
                           PRIVATE ${CMAKE_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR}/externals)

add_library(gpu_model model.cu)
set_target_properties(gpu_model PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(gpu_model quadutil ${NVTX_LIBRARY})
target_include_directories(gpu_model PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(test_cpu_vs_gpu test_cpu_vs_gpu.cu)
target_link_libraries(test_cpu_vs_gpu cpu_model gpu_model)
set_target_properties(test_cpu_vs_gpu PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_include_directories(test_cpu_vs_gpu PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_cpu_vs_gpu test_cpu_vs_gpu)


add_executable(hello hello.cu)
set_target_properties(hello PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
add_test(hello hello)

add_library(do_integration_from_c do_integration_from_c.cu)
set_target_properties(do_integration_from_c PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(do_integration_from_c util ${NVTX_LIBRARY})
target_include_directories(do_integration_from_c PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(test_integration_from_c test_integration_from_c.c)
target_link_libraries(test_integration_from_c do_integration_from_c)
add_test(test_integration_from_c test_integration_from_c)

add_executable(StatefullIntegrand StatefulIntegrand.cu)
set_target_properties(StatefullIntegrand PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(StatefullIntegrand quadutil ${NVTX_LIBRARY})
target_include_directories(StatefullIntegrand PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(StatefullIntegrand StatefullIntegrand)

add_executable(test_cuhre_convergence test_cuhre_convergence.cu)
set_target_properties(test_cuhre_convergence PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(test_cuhre_convergence util ${NVTX_LIBRARY})
target_include_directories(test_cuhre_convergence PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_cuhre_convergence test_cuhre_convergence)

add_executable(BoxIntegrals BoxIntegrals.cu)
set_target_properties(BoxIntegrals PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(BoxIntegrals util ${NVTX_LIBRARY})
target_include_directories(BoxIntegrals PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(BoxIntegrals BoxIntegrals)

add_executable(NanIntegral NanIntegral.cu)
set_target_properties(NanIntegral PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(NanIntegral util ${NVTX_LIBRARY})
target_include_directories(NanIntegral PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(NanIntegral NanIntegral)

add_executable(test_volume test_volume.cu)
set_target_properties(test_volume PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(test_volume util ${NVTX_LIBRARY})
target_include_directories(test_volume PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_volume test_volume)


add_executable(RegionSampling RegionSampling.cu)
set_target_properties(RegionSampling PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(RegionSampling util ${NVTX_LIBRARY})
target_include_directories(RegionSampling PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(RegionSampling RegionSampling)

add_executable(FourthDividedDifference FourthDividedDifference.cu)
set_target_properties(FourthDividedDifference PROPERTIES
  POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH}
  CUDA_ARCHITECTURES 70-real)
target_link_libraries(FourthDividedDifference util ${NVTX_LIBRARY})
target_include_directories(FourthDividedDifference PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(FourthDividedDifference FourthDividedDifference)

add_executable(Interpolation Interpolation.cu)
set_target_properties(Interpolation PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(Interpolation util ${NVTX_LIBRARY})
target_include_directories(Interpolation PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
  ${CMAKE_SOURCE_DIR}/tests/data
)
add_test(Interpolation Interpolation)


add_executable(Interpolation1D Interpolation1D.cu)
set_target_properties(Interpolation1D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(Interpolation1D quadutil ${NVTX_LIBRARY})
target_include_directories(Interpolation1D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Interpolation1D Interpolation1D)

add_executable(Interpolation2D Interpolation2D.cu)
set_target_properties(Interpolation2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(Interpolation2D quadutil ${NVTX_LIBRARY})
target_include_directories(Interpolation2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Interpolation2D Interpolation2D)
