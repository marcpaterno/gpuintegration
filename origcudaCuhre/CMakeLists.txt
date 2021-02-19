find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)

add_executable(origpdcuhre cudaCuhre.cu)
set_property(TARGET origpdcuhre PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(origpdcuhre PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(origpdcuhre ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(origpdcuhre PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
   ${CMAKE_SOURCE_DIR}/origcudaCuhre/quad/util/
)