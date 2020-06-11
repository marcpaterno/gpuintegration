
find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)


add_executable(fun6d_demo fun6d_demo.cc)
target_include_directories(fun6d_demo PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba/include/
  ${CMAKE_SOURCE_DIR}
  )
target_link_libraries(fun6d_demo PRIVATE ${CUBA_LIBRARIES})

add_executable(cuda_genz_1_8d genz_1_8d.cu)
target_compile_options(cuda_genz_1_8d PRIVATE "-arch=sm_70")
set_target_properties(cuda_genz_1_8d PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cuda_genz_1_8d util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(cuda_genz_1_8d PRIVATE
  ${CMAKE_SOURCE_DIR}
  )

add_executable(absCosSum5D absCosSum5D.cu)
target_compile_options(absCosSum5D PRIVATE "-arch=sm_70")
set_target_properties(absCosSum5D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(absCosSum5D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(absCosSum5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  )

add_executable(genz_1_8d genz_1_8d.cc)
target_include_directories(genz_1_8d PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba/include
  ${CMAKE_SOURCE_DIR}
  )
target_link_libraries(genz_1_8d PRIVATE ${CUBA_LIBRARIES})

