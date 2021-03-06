add_subdirectory(mcubes)
add_subdirectory(mcubes_1D)

add_executable(vegasGPU vegasGPU.cu)
target_compile_features(vegasGPU PUBLIC cxx_std_17)
target_compile_options(vegasGPU PRIVATE "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(vegasGPU PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(vegasGPU PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(vegasGPU PRIVATE     ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${CMAKE_SOURCE_DIR}/vegas
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")


add_executable(vegasCPU vegasCPU.cc)
set_target_properties(vegasCPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(vegasCPU PRIVATE ${CUBA_LIBRARIES} ${GSL_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(vegasCPU PRIVATE
			   ${CMAKE_SOURCE_DIR}
			   ${CMAKE_SOURCE_DIR}/externals
			   ${CMAKE_SOURCE_DIR}/vegas
			   ${EXTERNALS_DIR}
			   ${CUBA_INCLUDE_DIR}
			   ${GSL_INCLUDE_DIR}
			   ${CUBACPP_DIR}
			   "/opt/openmpi-wahab/include")

add_executable(Genz4_5D Genz4_5D.cu)
target_compile_features(Genz4_5D PUBLIC cxx_std_17)
target_compile_options(Genz4_5D PRIVATE "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(Genz4_5D PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(Genz4_5D PRIVATE     ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${CMAKE_SOURCE_DIR}/vegas
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")
                                                
add_executable(sigMiscent sigMiscent.cu)
target_compile_features(sigMiscent PUBLIC cxx_std_17)
target_compile_options(sigMiscent PRIVATE "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(sigMiscent PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(sigMiscent PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(sigMiscent PRIVATE     ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${CMAKE_SOURCE_DIR}/vegas
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")                                      