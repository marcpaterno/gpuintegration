add_executable(vegas_f0_6D f0.cu)
target_compile_features(vegas_f0_6D PUBLIC cxx_std_17)
target_compile_options(vegas_f0_6D PRIVATE "-O2" "-DCURAND" "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(vegas_f0_6D PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(vegas_f0_6D PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} models utils)
target_include_directories(vegas_f0_6D PRIVATE     ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${CMAKE_SOURCE_DIR}/vegas
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")
                                                
add_executable(vegas_f1_9D f1.cu)
target_compile_features(vegas_f1_9D PUBLIC cxx_std_17)
target_compile_options(vegas_f1_9D PRIVATE "-O2" "-DCURAND" "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(vegas_f1_9D PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(vegas_f1_9D PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} models utils)
target_include_directories(vegas_f1_9D PRIVATE     ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${CMAKE_SOURCE_DIR}/vegas
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")
 
add_executable(vegasB8_22 vegas_B8_22.cu)
target_compile_features(vegasB8_22 PUBLIC cxx_std_17)
target_compile_options(vegasB8_22 PRIVATE "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(vegasB8_22 PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(vegasB8_22 PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(vegasB8_22 PRIVATE     ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${CMAKE_SOURCE_DIR}/vegas
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")
 
add_executable(vegas_f2_2D f2.cu)
target_compile_features(vegas_f2_2D PUBLIC cxx_std_17)
target_compile_options(vegas_f2_2D PRIVATE "-O2" "-DCURAND" "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(vegas_f2_2D PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(vegas_f2_2D PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} models utils)
target_include_directories(vegas_f2_2D PRIVATE     ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${CMAKE_SOURCE_DIR}/vegas
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")                                                