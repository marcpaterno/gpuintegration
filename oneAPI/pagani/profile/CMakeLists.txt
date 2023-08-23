add_subdirectory(simple_funcs)
add_subdirectory(sum_of_args_integrands)

add_executable(oneapi_profile_SinSum oneapi_profile_SinSum.cpp)
target_include_directories(oneapi_profile_SinSum PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_SinSum PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_SinSum PRIVATE "-lonemkl")

add_executable(oneapi_profile_Genz3_3D oneapi_profile_Genz3_3D.cpp)
target_include_directories(oneapi_profile_Genz3_3D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_Genz3_3D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_Genz3_3D PRIVATE "-lonemkl")

add_executable(oneapi_profile_Genz3_8D oneapi_profile_Genz3_8D.cpp)
target_include_directories(oneapi_profile_Genz3_8D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_Genz3_8D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_Genz3_8D PRIVATE "-lonemkl")

add_executable(oneapi_profile_Genz2_6D oneapi_profile_Genz2_6D.cpp)
target_include_directories(oneapi_profile_Genz2_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_Genz2_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_Genz2_6D PRIVATE "-lonemkl")

add_executable(oneapi_profile_Genz4_5D oneapi_profile_Genz4_5D.cpp)
target_include_directories(oneapi_profile_Genz4_5D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_Genz4_5D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_Genz4_5D PRIVATE "-lonemkl")

add_executable(oneapi_profile_Genz4_6D oneapi_profile_Genz4_6D.cpp)
target_include_directories(oneapi_profile_Genz4_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_Genz4_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_Genz4_6D PRIVATE "-lonemkl")

add_executable(oneapi_profile_Genz5_8D oneapi_profile_Genz5_8D.cpp)
target_include_directories(oneapi_profile_Genz5_8D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_Genz5_8D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_Genz5_8D PRIVATE "-lonemkl")

add_executable(oneapi_profile_Genz6_6D oneapi_profile_Genz6_6D.cpp)
target_include_directories(oneapi_profile_Genz6_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_Genz6_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_Genz6_6D PRIVATE "-lonemkl")

add_executable(oneapi_execute_8D_benchmark_integrands_on_device execute_8D_benchmark_integrands_on_device.cpp)
target_include_directories(oneapi_execute_8D_benchmark_integrands_on_device PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_execute_8D_benchmark_integrands_on_device PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_execute_8D_benchmark_integrands_on_device PRIVATE "-lonemkl")

add_executable(oneapi_execute_7D_benchmark_integrands_on_device execute_7D_benchmark_integrands_on_device.cpp)
target_include_directories(oneapi_execute_7D_benchmark_integrands_on_device PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_execute_7D_benchmark_integrands_on_device PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_execute_7D_benchmark_integrands_on_device PRIVATE "-lonemkl")

add_executable(oneapi_execute_6D_benchmark_integrands_on_device execute_6D_benchmark_integrands_on_device.cpp)
target_include_directories(oneapi_execute_6D_benchmark_integrands_on_device PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_execute_6D_benchmark_integrands_on_device PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_execute_6D_benchmark_integrands_on_device PRIVATE "-lonemkl")

add_executable(oneapi_execute_5D_benchmark_integrands_on_device execute_5D_benchmark_integrands_on_device.cpp)
target_include_directories(oneapi_execute_5D_benchmark_integrands_on_device PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_execute_5D_benchmark_integrands_on_device PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_execute_5D_benchmark_integrands_on_device PRIVATE "-lonemkl")


add_executable(oneapi_execute_math_functions_on_device execute_math_functions_on_device.cpp)
target_include_directories(oneapi_execute_math_functions_on_device PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_execute_math_functions_on_device PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_execute_math_functions_on_device PRIVATE "-lonemkl")

add_executable(oneapi_profile_integrands profile_integrands.cpp)
target_include_directories(oneapi_profile_integrands PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_integrands PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_integrands PRIVATE "-lonemkl" "-g" "-Xcuda-ptxas -v")

add_executable(oneapi_profile_mcubes_integrands profile_mcubes_integrands.cpp)
target_include_directories(oneapi_profile_mcubes_integrands PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_mcubes_integrands PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_mcubes_integrands PRIVATE "-lonemkl")

add_executable(oneapi_profile_pagani_integrands profile_pagani_integrands.cpp)
target_include_directories(oneapi_profile_pagani_integrands PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_pagani_integrands PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_pagani_integrands PRIVATE "-lonemkl")

add_executable(oneapi_atomic_addition atomic_addition.cpp)
target_include_directories(oneapi_atomic_addition PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_atomic_addition PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_atomic_addition PRIVATE "-lonemkl")
