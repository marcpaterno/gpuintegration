find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)

add_executable(cudaCuhre cudaCuhre.cu)
set_property(TARGET cudaCuhre PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(cudaCuhre PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cudaCuhre quadutil ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

add_executable(test_cuhre_convergence test_cuhre_convergence.cu)
set_property(TARGET test_cuhre_convergence PROPERTY CUDA_ARCHITECTURES 70-real)
#target_compile_options(test_cuhre_convergence PRIVATE "-arch=sm_70")
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
add_test(Genz3_3DGPU Genz3_3DGPU)

add_executable(Genz3_6DGPU Genz3_6D.cu)
target_compile_options(Genz3_6DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz3_6DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz3_6DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz3_6DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Genz3_6DGPU Genz3_3DGPU)

add_executable(Genz3_8DGPU Genz3_8D.cu)
target_compile_options(Genz3_8DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz3_8DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz3_8DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz3_8DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Genz3_8DGPU Genz3_3DGPU)

add_executable(Genz6_6DGPU Genz6_6D.cu)
target_compile_options(Genz6_6DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz6_6DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz6_6DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz6_6DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Genz6_6DGPU Genz3_3DGPU)

add_executable(Genz6_2DGPU Genz6_2D.cu)
target_compile_options(Genz6_2DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz6_2DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz6_2DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz6_2DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Genz6_2DGPU Genz3_3DGPU)

add_executable(fun6d_demo fun6d_demo.cu)
target_include_directories(fun6d_demo PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(fun6d_demo PRIVATE ${CUBA_LIBRARIES})
set_property(TARGET fun6d_demo PROPERTY CUDA_ARCHITECTURES 70-real)

add_executable(B_22cpu B_22cpu.cc)
target_include_directories(B_22cpu PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(B_22cpu PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz5_2DCPU Genz5_2D.cc)
target_include_directories(Genz5_2DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz5_2DCPU PRIVATE ${CUBA_LIBRARIES})

add_executable(Gauss9Dcpu Gauss9Dcpu.cc)
target_include_directories(Gauss9Dcpu PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Gauss9Dcpu PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz3_3DCPU Genz3_3D.cc)
target_include_directories(Genz3_3DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz3_3DCPU PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz3_6CPU Genz3_6D.cc)
target_include_directories(Genz3_6CPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz3_6CPU PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz3_8DCPU Genz3_8D.cc)
target_include_directories(Genz3_8DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz3_8DCPU PRIVATE ${CUBA_LIBRARIES})

add_executable(B_15cpu B_15cpu.cc)
target_include_directories(B_15cpu PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(B_15cpu PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz6_6DCPU Genz6_6D.cc)
target_include_directories(Genz6_6DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz6_6DCPU PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz6_2DCPU Genz6_2D.cc)
target_include_directories(Genz6_2DCPU PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(Genz6_2DCPU PRIVATE ${CUBA_LIBRARIES})

add_executable(absCosSum5Dplus1cpu absCosSum5Dplus1cpu.cc)
target_include_directories(absCosSum5Dplus1cpu PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(absCosSum5Dplus1cpu PRIVATE ${CUBA_LIBRARIES})

add_executable(absCosSum5Dcpu absCosSum5Dcpu.cc)
target_include_directories(absCosSum5Dcpu PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(absCosSum5Dcpu PRIVATE ${CUBA_LIBRARIES})

add_executable(cuda_genz_1_8d genz_1_8d.cu)
target_compile_options(cuda_genz_1_8d PRIVATE "-lineinfo")
set_property(TARGET cuda_genz_1_8d PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(cuda_genz_1_8d PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cuda_genz_1_8d util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(cuda_genz_1_8d PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(absCosSum5D absCosSum5D.cu)
target_compile_options(absCosSum5D PRIVATE "-lineinfo")
set_property(TARGET absCosSum5D PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(absCosSum5D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(absCosSum5D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(absCosSum5D PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(cuda_genz_1abs_5d genz_1abs_5d.cu)
set_property(TARGET cuda_genz_1abs_5d PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(cuda_genz_1abs_5d PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cuda_genz_1abs_5d util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(cuda_genz_1abs_5d PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(SinSum6Dcpu SinSum6D.cc)
target_include_directories(SinSum6Dcpu PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(SinSum6Dcpu PRIVATE ${CUBA_LIBRARIES})

add_executable(SinSum6gpu SinSum6D.cu)
target_compile_options(SinSum6gpu PRIVATE "-arch=sm_70")
set_target_properties(SinSum6gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SinSum6gpu util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(SinSum6gpu PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(genz_1_8d genz_1_8d.cc)
target_include_directories(genz_1_8d PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(genz_1_8d PRIVATE ${CUBA_LIBRARIES})

add_executable(genz_1abs_5d genz_1abs_5d.cc)
target_include_directories(genz_1abs_5d PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(genz_1abs_5d PRIVATE ${CUBA_LIBRARIES})
