add_executable(oneapi_pagani_genz_integrals genz_integrals.cpp)
target_include_directories(oneapi_pagani_genz_integrals PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_pagani_genz_integrals PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_pagani_genz_integrals PRIVATE "-lonemkl")

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

add_executable(header_inclusion_1 header_inclusion_1.cpp)
target_include_directories(header_inclusion_1 PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(header_inclusion_1 PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(header_inclusion_1 PRIVATE "-lonemkl")

add_executable(header_inclusion_2 header_inclusion_2.cpp)
target_include_directories(header_inclusion_2 PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(header_inclusion_2 PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(header_inclusion_2 PRIVATE "-lonemkl")
