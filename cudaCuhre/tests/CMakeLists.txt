find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)

add_executable(fun6d_demo fun6d_demo.cc)
target_include_directories(fun6d_demo PRIVATE
  ${CMAKE_SOURCE_DIR}/../cubacpp
  ${CMAKE_SOURCE_DIR}/../cuba
  ${CMAKE_SOURCE_DIR}
  )
target_link_libraries(fun6d_demo PRIVATE ${CUBA_LIBRARIES})

