add_executable(sigma_miscent_y1_scalarintegrand_test sigma_miscent_y1_scalarintegrand_test.cc)
target_link_libraries(sigma_miscent_y1_scalarintegrand_test PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(sigma_miscent_y1_scalarintegrand_test
                           PRIVATE ${CMAKE_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR}/externals
                                   ${EXTERNALS_DIR}
                                   ${CUBA_INCLUDE_DIR}
                                   ${CUBACPP_DIR}
                                   ${GSL_INCLUDE_DIR}
                                   ${EIGEN3_INCLUDE_DIR}
								   ${TOPDIR})
add_test(sigma_miscent_y1_scalarintegrand_test sigma_miscent_y1_scalarintegrand_test)
message(${GSL_LIBRARIES})
message(${GSL_INCLUDE_DIRS})
message(${GSL_LIBRARY})
#message(${GSL_CLBAS_LIBRARY})
add_executable(cudasInterp cudasInterp.cu)
target_compile_features(cudasInterp PUBLIC cxx_std_17)
target_compile_options(cudasInterp PRIVATE "-arch=sm_70" "-lineinfo" "--expt-relaxed-constexpr")
set_target_properties(cudasInterp PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cudasInterp PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(cudasInterp PRIVATE  ${CMAKE_SOURCE_DIR}
												${CMAKE_SOURCE_DIR}/externals
												${EXTERNALS_DIR}
												${CUBA_INCLUDE_DIR}
												${GSL_INCLUDE_DIR}
												${CUBACPP_DIR}
												"/opt/openmpi-wahab/include")
add_test(cudasInterp cudasInterp)

add_executable(test_io test_io.test.cc)
target_link_libraries(test_io PRIVATE models utils testmain ${GSL_LIBRARIES})
target_include_directories(test_io PRIVATE
                          ${CMAKE_SOURCE_DIR}
                          ${GSL_INCLUDE_DIR}
                          ${CMAKE_SOURCE_DIR}/externals)
add_test(test_io test_io)

