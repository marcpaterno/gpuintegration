add_executable(new_interface_Genz2_6D new_interface_Genz2_6D.cpp)
target_compile_options(new_interface_Genz2_6D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb")
#target_link_libraries(new_interface_Genz2_6D PRIVATE MKL::MKL_DPCPP)

#add_executable(new_interface_Genz3_3D Genz3_3D.cpp)
#target_link_libraries(new_interface_Genz3_3D PRIVATE MKL::MKL_DPCPP)

#add_executable(new_interface_Genz4_5D new_interface_Genz4_5D.dp.cpp)
#target_link_libraries(new_interface_Genz4_5D PRIVATE MKL::MKL_DPCPP)

#add_executable(new_interface_Genz5_8D new_interface_Genz5_8D.cpp)
#target_link_libraries(new_interface_Genz5_8D PRIVATE MKL::MKL_DPCPP)

#add_executable(new_interface_Genz6_6D new_interface_Genz6_6D.cpp)
#target_link_libraries(new_interface_Genz6_6D PRIVATE MKL::MKL_DPCPP)


