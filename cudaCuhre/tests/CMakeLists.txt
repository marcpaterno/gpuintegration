find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)

add_executable(hello hello.cu)
set_property(TARGET hello PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(hello PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_test(hello hello)

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

