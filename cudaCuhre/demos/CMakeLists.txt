add_subdirectory(Heuristics)
find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)
find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(cudaCuhre cudaCuhre.cu)
set_property(TARGET cudaCuhre PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(cudaCuhre PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cudaCuhre quadutil ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})

add_executable(B8_15 B8_15.cu)
target_compile_options(B8_15 PRIVATE "-arch=sm_70")
set_target_properties(B8_15 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(B8_15 util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(B8_15 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(DiagonalRidge2D diagonalRidge2D.cu)
target_compile_options(DiagonalRidge2D PRIVATE "-arch=sm_70" "-lineinfo")
set_target_properties(DiagonalRidge2D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(DiagonalRidge2D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(DiagonalRidge2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(B8_22 B8_22.cu)
target_compile_options(B8_22 PRIVATE "-arch=sm_70")
set_target_properties(B8_22 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(B8_22 util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(B8_22 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Gauss9D Gauss9D.cu)
target_compile_options(Gauss9D PRIVATE "-arch=sm_70")
set_target_properties(Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Gauss9D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Gauss9D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(absCosSum5Dplus1 absCosSum5Dplus1.cu)
target_compile_options(absCosSum5Dplus1 PRIVATE "-arch=sm_70")
set_target_properties(absCosSum5Dplus1 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(absCosSum5Dplus1 util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(absCosSum5Dplus1 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(Genz2_2DGPU Genz2_2D.cu)
target_compile_options(Genz2_2DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz2_2DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz2_2DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz2_2DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz2_6DGPU Genz2_6D.cu)
target_compile_options(Genz2_6DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz2_6DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz2_6DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz2_6DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz2_8DGPU Genz2_8D.cu)
target_compile_options(Genz2_8DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz2_8DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz2_8DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz2_8DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz3_3DGPU Genz3_3D.cu)
target_compile_options(Genz3_3DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz3_3DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz3_3DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz3_3DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz3_6DGPU Genz3_6D.cu)
target_compile_options(Genz3_6DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz3_6DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz3_6DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz3_6DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz3_8DGPU Genz3_8D.cu)
target_compile_options(Genz3_8DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz3_8DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz3_8DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz3_8DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(Genz4_5DGPU Genz4_5D.cu)
target_compile_options(Genz4_5DGPU PRIVATE "-arch=sm_70" "-lineinfo")
set_target_properties(Genz4_5DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz4_5DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain) 
target_include_directories(Genz4_5DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz4_8DGPU Genz4_8D.cu)
target_compile_options(Genz4_8DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz4_8DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz4_8DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz4_8DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(Genz5_2DGPU Genz5_2D.cu)
target_compile_options(Genz5_2DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz5_2DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz5_2DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz5_2DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz5_5DGPU Genz5_5D.cu)
target_compile_options(Genz5_5DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz5_5DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz5_5DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz5_5DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz5_8DGPU Genz5_8D.cu)
target_compile_options(Genz5_8DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz5_8DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz5_8DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz5_8DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(Genz6_2DGPU Genz6_2D.cu)
target_compile_options(Genz6_2DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz6_2DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz6_2DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz6_2DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz6_6DGPU Genz6_6D.cu)
target_compile_options(Genz6_6DGPU PRIVATE "-arch=sm_70")
set_target_properties(Genz6_6DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz6_6DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz6_6DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(fun6d_demo fun6d_demo.cu)
target_include_directories(fun6d_demo PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(fun6d_demo PRIVATE ${CUBA_LIBRARIES})
set_property(TARGET fun6d_demo PROPERTY CUDA_ARCHITECTURES 70-real)

add_executable(Genz1_8DGPU Genz1_8D.cu)
target_compile_options(Genz1_8DGPU PRIVATE "-lineinfo")
set_property(TARGET Genz1_8DGPU PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(Genz1_8DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz1_8DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(Genz1_8DGPU PRIVATE
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

add_executable(SinSum6DGPU SinSum6D.cu)
target_compile_options(SinSum6DGPU PRIVATE "-arch=sm_70")
set_target_properties(SinSum6DGPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SinSum6DGPU util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(SinSum6DGPU PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(sigMiscentGPU sigMiscent.cu)
target_compile_features(sigMiscentGPU PUBLIC cxx_std_17)
target_compile_options(sigMiscentGPU PRIVATE "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(sigMiscentGPU PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(sigMiscentGPU PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(sigMiscentGPU PRIVATE  ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")

add_executable(scalarIntegrandSigmaGPU scalarIntegrandSigma.cu)
target_compile_features(scalarIntegrandSigmaGPU PUBLIC cxx_std_17)
target_compile_options(scalarIntegrandSigmaGPU PRIVATE "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(scalarIntegrandSigmaGPU PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(scalarIntegrandSigmaGPU PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(scalarIntegrandSigmaGPU PRIVATE  ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")
												
add_executable(sigMiscentCPU sigMiscent.cc)
target_link_libraries(sigMiscentCPU PRIVATE ${CUBA_LIBRARIES} ${GSL_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(sigMiscentCPU
                           PRIVATE ${CMAKE_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR}/externals
                                   ${EXTERNALS_DIR}
                                   ${CUBA_INCLUDE_DIR}
                                   ${CUBACPP_DIR}
                                   ${GSL_INCLUDE_DIR}
                                   ${EIGEN3_INCLUDE_DIR}
								   ${TOPDIR})

																					
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

add_executable(Genz5_5DCPU Genz5_5D.cc)
target_include_directories(Genz5_5DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz5_5DCPU PRIVATE ${CUBA_LIBRARIES})

add_executable(Gauss9Dcpu Gauss9Dcpu.cc)
target_include_directories(Gauss9Dcpu PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Gauss9Dcpu PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz6_2DCPU Genz6_2D.cc)
target_include_directories(Genz6_2DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz6_2DCPU PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz2_6DCPU Genz2_6D.cc)
target_include_directories(Genz2_6DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz2_6DCPU PRIVATE ${CUBA_LIBRARIES})

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

add_executable(Genz4_5DCPU Genz4_5D.cc)
target_include_directories(Genz4_5DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz4_5DCPU PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz5_8DCPU Genz5_8D.cc)
target_include_directories(Genz5_8DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz5_8DCPU PRIVATE ${CUBA_LIBRARIES})

add_executable(Genz6_6DCPU Genz6_6D.cc)
target_include_directories(Genz6_6DCPU PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CUBA_INCLUDE_DIR}
)
target_link_libraries(Genz6_6DCPU PRIVATE ${CUBA_LIBRARIES})

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

add_executable(SinSum6Dcpu SinSum6D.cc)
target_include_directories(SinSum6Dcpu PRIVATE
  ${CUBA_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}
)
target_link_libraries(SinSum6Dcpu PRIVATE ${CUBA_LIBRARIES})


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
