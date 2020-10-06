find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)

add_executable(test_cuhre_convergence test_cuhre_convergence.cu)
set_property(TARGET test_cuhre_convergence PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(test_cuhre_convergence PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(test_cuhre_convergence util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(test_cuhre_convergence PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_cuhre_convergence test_cuhre_convergence)

add_executable(BoxIntegrals BoxIntegrals.cu)
set_property(TARGET BoxIntegrals PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(BoxIntegrals PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(BoxIntegrals util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(BoxIntegrals PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(BoxIntegrals BoxIntegrals)

add_executable(test_volume test_volume.cu)
target_compile_options(test_volume PRIVATE "-arch=sm_70")
set_target_properties(test_volume PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(test_volume util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(test_volume PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_volume test_volume)

add_executable(B8_22 B8_22.cu)
set_target_properties(B8_22 PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_property(TARGET B8_22 PROPERTY CUDA_ARCHITECTURES 70-real)
target_link_libraries(B8_22 util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(B8_22 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(B8_22 B8_22)

add_executable(Gauss9D Gauss9D.cu)
target_compile_options(Gauss9D PRIVATE "-arch=sm_70")
set_target_properties(Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Gauss9D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Gauss9D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(B8_22 Gauss9D)

add_executable(absCosSum5Dplus1 absCosSum5Dplus1.cu)
target_compile_options(absCosSum5Dplus1 PRIVATE "-arch=sm_70")
set_target_properties(absCosSum5Dplus1 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(absCosSum5Dplus1 util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(absCosSum5Dplus1 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(absCosSum5Dplus1 absCosSum5Dplus1)

add_executable(B8_15 B8_15.cu)
target_compile_options(B8_15 PRIVATE "-arch=sm_70")
set_target_properties(B8_15 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(B8_15 util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(B8_15 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(B8_15 B8_15)

add_executable(Genz5_2DGPU Genz5_2D.cu)
target_compile_options(Genz5_2DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz5_2DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz5_2DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz5_2DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Genz5_2DGPU Genz5_2DGPU)

add_executable(Genz3_3DGPU Genz3_3D.cu)
target_compile_options(Genz3_3DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz3_3DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz3_3DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz3_3DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
