add_library(testmain SHARED main.cc)
target_include_directories(testmain
                           PUBLIC ${CMAKE_SOURCE_DIR}
                                  ${CMAKE_SOURCE_DIR}/externals)

add_executable(sigma_miscent_y1_scalarintegrand_test sigma_miscent_y1_scalarintegrand_test.cc)
target_link_libraries(sigma_miscent_y1_scalarintegrand_test PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} sigma_miscent_y1_scalarintegrand models utils)
target_include_directories(sigma_miscent_y1_scalarintegrand_test
                           PRIVATE ${CMAKE_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR}/externals
                                   ${EXTERNALS_DIR}
                                   ${CUBA_INCLUDE_DIR}
                                   ${CUBACPP_DIR}
                                   ${GSL_INCLUDE_DIR}
                                   ${EIGEN3_INCLUDE_DIR})
add_test(sigma_miscent_y1_scalarintegrand_test sigma_miscent_y1_scalarintegrand_test)

add_executable(test_io test_io.test.cc)
target_link_libraries(test_io PRIVATE models utils testmain ${GSL_LIBRARIES})
target_include_directories(test_io PRIVATE
                          ${CMAKE_SOURCE_DIR}
                          ${GSL_INCLUDE_DIR}
                          ${CMAKE_SOURCE_DIR}/externals)
add_test(test_io test_io)

