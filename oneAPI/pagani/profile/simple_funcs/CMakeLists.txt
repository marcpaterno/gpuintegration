add_executable(oneapi_profile_3D oneapi_profile_3D.cpp)
target_include_directories(oneapi_profile_3D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_3D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_3D PRIVATE "-lonemkl")

add_executable(oneapi_profile_5D oneapi_profile_5D.cpp)
target_include_directories(oneapi_profile_5D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_5D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_5D PRIVATE "-lonemkl")

add_executable(oneapi_profile_6D oneapi_profile_6D.cpp)
target_include_directories(oneapi_profile_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_6D PRIVATE "-lonemkl")

add_executable(oneapi_profile_8D oneapi_profile_8D.cpp)
target_include_directories(oneapi_profile_8D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_8D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_8D PRIVATE "-lonemkl")


