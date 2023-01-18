add_executable(oneapi_new_interface_Genz2_6D new_interface_Genz2_6D.cpp)
target_include_directories(oneapi_new_interface_Genz2_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_new_interface_Genz2_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_new_interface_Genz2_6D PRIVATE "-lonemkl")

add_executable(oneapi_new_interface_Genz3_3D new_interface_Genz3_3D.cpp)
target_include_directories(oneapi_new_interface_Genz3_3D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_new_interface_Genz3_3D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_new_interface_Genz3_3D PRIVATE "-lonemkl")

add_executable(oneapi_new_interface_Genz4_5D new_interface_Genz4_5D.cpp)
target_include_directories(oneapi_new_interface_Genz4_5D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_new_interface_Genz4_5D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_new_interface_Genz4_5D PRIVATE "-lonemkl")

add_executable(oneapi_new_interface_Genz5_8D new_interface_Genz5_8D.cpp)
target_include_directories(oneapi_new_interface_Genz5_8D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_new_interface_Genz5_8D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_new_interface_Genz5_8D PRIVATE "-lonemkl")

add_executable(oneapi_new_interface_Genz6_6D new_interface_Genz6_6D.cpp)
target_include_directories(oneapi_new_interface_Genz6_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_new_interface_Genz6_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_new_interface_Genz6_6D PRIVATE "-lonemkl")
