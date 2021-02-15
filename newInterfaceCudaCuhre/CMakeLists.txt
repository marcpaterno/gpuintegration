find_package(CUBA REQUIRED)
find_package(GSL REQUIRED)

add_executable(origpdcuhreUpdated cudaCuhre.cu)
target_compile_options(origpdcuhreUpdated PRIVATE "-arch=sm_70" "-g" "-lineinfo")
set_property(TARGET origpdcuhreUpdated PROPERTY CUDA_ARCHITECTURES 70-real)
set_target_properties(origpdcuhreUpdated PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(origpdcuhreUpdated ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
target_include_directories(origpdcuhreUpdated PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
   ${CMAKE_SOURCE_DIR}/origcudaCuhre/quad/util/
)