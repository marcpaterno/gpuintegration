find_package(CUBA REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)

add_executable(fun6d_test fun6d_test.cc)
target_include_directories(fun6d_test PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba
  ${CMAKE_SOURCE_DIR}/../eigen
  ${CMAKE_SOURCE_DIR}
  )
target_link_libraries(fun6d_test PRIVATE ${CUBA_LIBRARIES})

