find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)

add_executable(test_cuhre_convergence test_cuhre_convergence.cu)
target_compile_options(test_cuhre_convergence PRIVATE "-arch=sm_70")
set_target_properties(test_cuhre_convergence PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(test_cuhre_convergence util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(test_cuhre_convergence PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_cuhre_convergence test_cuhre_convergence)

add_executable(BoxIntegrals BoxIntegrals.cu)
target_compile_options(BoxIntegrals PRIVATE "-arch=sm_70")
set_target_properties(BoxIntegrals PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(BoxIntegrals util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(BoxIntegrals PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(BoxIntegrals BoxIntegrals)

add_executable(B8_22 B8_22.cu)
target_compile_options(B8_22 PRIVATE "-arch=sm_70")
set_target_properties(B8_22 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(B8_22 util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(B8_22 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(B8_22 B8_22)

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

add_executable(fun6d_demo fun6d_demo.cc)
target_include_directories(fun6d_demo PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba/include/
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(fun6d_demo PRIVATE ${CUBA_LIBRARIES})

add_executable(B_22cpu B_22cpu.cc)
target_include_directories(B_22cpu PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba/include/
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(B_22cpu PRIVATE ${CUBA_LIBRARIES})

add_executable(B_15cpu B_15cpu.cc)
target_include_directories(B_15cpu PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba/include/
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(B_15cpu PRIVATE ${CUBA_LIBRARIES})

add_executable(absCosSum5Dplus1cpu absCosSum5Dplus1cpu.cc)
target_include_directories(absCosSum5Dplus1cpu PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba/include/
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(absCosSum5Dplus1cpu PRIVATE ${CUBA_LIBRARIES})


add_executable(cuda_genz_1_8d genz_1_8d.cu)
target_compile_options(cuda_genz_1_8d PRIVATE "-arch=sm_70" "-lineinfo")
set_target_properties(cuda_genz_1_8d PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cuda_genz_1_8d util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(cuda_genz_1_8d PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(absCosSum5D absCosSum5D.cu)
target_compile_options(absCosSum5D PRIVATE "-arch=sm_70" "-lineinfo")
set_target_properties(absCosSum5D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(absCosSum5D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(absCosSum5D PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(cuda_genz_1abs_5d genz_1abs_5d.cu)
target_compile_options(cuda_genz_1abs_5d PRIVATE "-arch=sm_70")
set_target_properties(cuda_genz_1abs_5d PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cuda_genz_1abs_5d util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(cuda_genz_1abs_5d PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(genz_1_8d genz_1_8d.cc)
target_include_directories(genz_1_8d PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba/include
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(genz_1_8d PRIVATE ${CUBA_LIBRARIES})

add_executable(genz_1abs_5d genz_1abs_5d.cc)
target_include_directories(genz_1abs_5d PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba/include
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(genz_1abs_5d PRIVATE ${CUBA_LIBRARIES})
