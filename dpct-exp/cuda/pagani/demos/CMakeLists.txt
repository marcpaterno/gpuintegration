add_executable(dpct_oneapi_new_interface_Genz6_6D new_interface_Genz6_6D.dp.cpp)
target_include_directories(dpct_oneapi_new_interface_Genz6_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(dpct_oneapi_new_interface_Genz6_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(dpct_oneapi_new_interface_Genz6_6D PRIVATE "-lonemkl")

add_executable(dpct_repeat_profile_Genz6_6D dpct_repeat_profile_Genz6_6D.cpp)
target_include_directories(dpct_repeat_profile_Genz6_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(dpct_repeat_profile_Genz6_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(dpct_repeat_profile_Genz6_6D PRIVATE "-lonemkl")

add_executable(dpct_repeat_profile_Genz2_6D dpct_repeat_profile_Genz2_6D.cpp)
target_include_directories(dpct_repeat_profile_Genz2_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(dpct_repeat_profile_Genz2_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(dpct_repeat_profile_Genz2_6D PRIVATE "-lonemkl")
