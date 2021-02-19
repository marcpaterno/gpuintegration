find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)

add_executable(Genz1_8Dpdc Genz1_8D.cu)
target_compile_options(Genz1_8Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz1_8Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz1_8Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz1_8Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(B8_15pdc B8_15.cu)
target_compile_options(B8_15pdc PRIVATE "-arch=sm_70")
set_target_properties(B8_15pdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(B8_15pdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(B8_15pdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(DiagonalRidge2Dpdc diagonalRidge2D.cu)
target_compile_options(DiagonalRidge2Dpdc PRIVATE "-arch=sm_70")
set_target_properties(DiagonalRidge2Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(DiagonalRidge2Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(DiagonalRidge2Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(B8_22pdc B8_22.cu)
target_compile_options(B8_22pdc PRIVATE "-arch=sm_70")
set_target_properties(B8_22pdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(B8_22pdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(B8_22pdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Gauss9Dpdc Gauss9D.cu)
target_compile_options(Gauss9Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Gauss9Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Gauss9Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Gauss9Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz2_2Dpdc Genz2_2D.cu)
target_compile_options(Genz2_2Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz2_2Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz2_2Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz2_2Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz2_6Dpdc Genz2_6D.cu)
target_compile_options(Genz2_6Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz2_6Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz2_6Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz2_6Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz2_8Dpdc Genz2_8D.cu)
target_compile_options(Genz2_8Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz2_8Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz2_8Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz2_8Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz3_3Dpdc Genz3_3D.cu)
target_compile_options(Genz3_3Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz3_3Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz3_3Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz3_3Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz3_6Dpdc Genz3_6D.cu)
target_compile_options(Genz3_6Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz3_6Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz3_6Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz3_6Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz3_8Dpdc Genz3_8D.cu)
target_compile_options(Genz3_8Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz3_8Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz3_8Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain )
target_include_directories(Genz3_8Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz4_5Dpdc Genz4_5D.cu)
target_compile_options(Genz4_5Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz4_5Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz4_5Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain) 
target_include_directories(Genz4_5Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz4_8Dpdc Genz4_8D.cu)
target_compile_options(Genz4_8Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz4_8Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz4_8Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz4_8Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(Genz5_2Dpdc Genz5_2D.cu)
target_compile_options(Genz5_2Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz5_2Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz5_2Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz5_2Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz5_5Dpdc Genz5_5D.cu)
target_compile_options(Genz5_5Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz5_5Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz5_5Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz5_5Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz5_8Dpdc Genz5_8D.cu)
target_compile_options(Genz5_8Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz5_8Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz5_8Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz5_8Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(Genz6_2Dpdc Genz6_2D.cu)
target_compile_options(Genz6_2Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz6_2Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz6_2Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz6_2Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Genz6_6Dpdc Genz6_6D.cu)
target_compile_options(Genz6_6Dpdc PRIVATE "-arch=sm_70")
set_target_properties(Genz6_6Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(Genz6_6Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(Genz6_6Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(SinSum6Dpdc SinSum6D.cu)
target_compile_options(SinSum6Dpdc PRIVATE "-arch=sm_70")
set_target_properties(SinSum6Dpdc PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(SinSum6Dpdc util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(SinSum6Dpdc PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(sigMiscentpdc sigMiscent.cu)
target_compile_features(sigMiscentpdc PUBLIC cxx_std_17)
target_compile_options(sigMiscentpdc PRIVATE "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(sigMiscentpdc PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(sigMiscentpdc PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(sigMiscentpdc PRIVATE  ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")

add_executable(scalarIntegrandSigmapdc scalarIntegrandSigma.cu)
target_compile_features(scalarIntegrandSigmapdc PUBLIC cxx_std_17)
target_compile_options(scalarIntegrandSigmapdc PRIVATE "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(scalarIntegrandSigmapdc PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES  70-real)
target_link_libraries(scalarIntegrandSigmapdc PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(scalarIntegrandSigmapdc PRIVATE  ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")
												

																				