add_executable(new_interface_Genz4_5D new_interface_Genz4_5D.dp.cpp)
target_link_libraries(new_interface_Genz4_5D PRIVATE MKL::MKL_DPCPP)

#set_target_properties(new_interface_Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
#target_link_libraries(new_interface_Genz4_5D ${NVTX_LIBRARY})
#target_include_directories(new_interface_Genz4_5D PRIVATE
#  ${CMAKE_SOURCE_DIR}
#  ${CMAKE_SOURCE_DIR}/externals
#)

add_executable(new_interface_Genz3_3D Genz3_3D.cpp)
target_link_libraries(new_interface_Genz3_3D PRIVATE MKL::MKL_DPCPP)
