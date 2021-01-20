add_executable(hGenz3_6D Genz3_6D.cu)
target_compile_options(hGenz3_6D PRIVATE "-arch=sm_70")
set_target_properties(hGenz3_6D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hGenz3_6D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(hGenz3_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(hGenz3_8D Genz3_8D.cu)
target_compile_options(hGenz3_8D PRIVATE "-arch=sm_70")
set_target_properties(hGenz3_8D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hGenz3_8D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(hGenz3_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(hGenz4_5D Genz4_5D.cu)
target_compile_options(hGenz4_5D PRIVATE "-arch=sm_70")
set_target_properties(hGenz4_5D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hGenz4_5D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(hGenz4_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(hGenz5_5D Genz5_5D.cu)
target_compile_options(hGenz5_5D PRIVATE "-arch=sm_70")
set_target_properties(hGenz5_5D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hGenz5_5D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(hGenz5_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(hGenz5_8D Genz5_8D.cu)
target_compile_options(hGenz5_8D PRIVATE "-arch=sm_70")
set_target_properties(hGenz5_8D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hGenz5_8D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(hGenz5_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(hGenz6_6D Genz6_6D.cu)
target_compile_options(hGenz6_6D PRIVATE "-arch=sm_70")
set_target_properties(hGenz6_6D PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hGenz6_6D util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(hGenz6_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(hB8_15 B8_15.cu)
target_compile_options(hB8_15 PRIVATE "-arch=sm_70")
set_target_properties(hB8_15 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hB8_15 util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(hB8_15 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(hB8_22 B8_22.cu)
target_compile_options(hB8_22 PRIVATE "-arch=sm_70")
set_target_properties(hB8_22 PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hB8_22 util ${MPI_CXX_LIBRARIES} ${OpenMP_CXX_LIBRARIES} testmain)
target_include_directories(hB8_22 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)