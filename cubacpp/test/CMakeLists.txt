add_library(testmain SHARED main.cc)
target_include_directories(testmain
                           PUBLIC ${CMAKE_SOURCE_DIR}
                                  ${CMAKE_SOURCE_DIR}/externals)

add_executable(integration_volume_test integration_volume.test.cc)
target_link_libraries(integration_volume_test PRIVATE testmain ${CUBA_LIBRARIES})
target_include_directories(integration_volume_test
                           PRIVATE ${CMAKE_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR}/externals
                                   ${CUBA_INCLUDE_DIR}
                                   ${GSL_INCLUDE_DIR}
                                   )
add_test(integration_volume_test integration_volume_test)

add_executable(integrand_traits_test integrand_traits.test.cc)
target_link_libraries(integrand_traits_test PRIVATE testmain)
target_include_directories(integrand_traits_test
                           PRIVATE ${CMAKE_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR}/externals)
add_test(integrand_traits_test integrand_traits_test)

add_executable(integrand_test integrand.test.cc)
target_link_libraries(integrand_test PRIVATE testmain ${CUBA_LIBRARIES})
target_include_directories(integrand_test
                           PRIVATE ${CMAKE_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR}/externals
                                   ${CUBA_INCLUDE_DIR}
                                   ${GSL_INCLUDE_DIR}
                                   )
add_test(integrand_test integrand_test)

add_executable(integration_test integration.test.cc)
target_link_libraries(integration_test
                      PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES})
target_include_directories(integration_test
                           PRIVATE ${CMAKE_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR}/externals
                                   ${CUBA_INCLUDE_DIR}
                                   ${GSL_INCLUDE_DIR}
                                   )
add_test(integration_test integration_test)

add_executable(two_d_plot two_d_plot.cc)
target_link_libraries(two_d_plot PRIVATE ${CUBA_LIBRARIES})
target_include_directories(two_d_plot PRIVATE ${CMAKE_SOURCE_DIR} ${CUBA_INCLUDE_DIR})

add_executable(performance_comparison performance_comparison.cc)
target_link_libraries(performance_comparison PRIVATE ${CUBA_LIBRARIES})
target_include_directories(performance_comparison PRIVATE ${CMAKE_SOURCE_DIR} ${CUBA_INCLUDE_DIR})

add_executable(performance_comparison2 performance_comparison2.cc)
target_link_libraries(performance_comparison2 PRIVATE ${CUBA_LIBRARIES})
target_include_directories(performance_comparison2 PRIVATE ${CMAKE_SOURCE_DIR} ${CUBA_INCLUDE_DIR})