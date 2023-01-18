add_executable(oneapi_profile_pagani_kernel_f_3D oneapi_profile_pagani_kernel_f_3D.cpp)
target_include_directories(oneapi_profile_pagani_kernel_f_3D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_pagani_kernel_f_3D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_pagani_kernel_f_3D PRIVATE "-lonemkl")

add_executable(oneapi_profile_pagani_kernel_f_5D oneapi_profile_pagani_kernel_f_5D.cpp)
target_include_directories(oneapi_profile_pagani_kernel_f_5D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_pagani_kernel_f_5D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_pagani_kernel_f_5D PRIVATE "-lonemkl")

add_executable(oneapi_profile_pagani_kernel_f_6D oneapi_profile_pagani_kernel_f_6D.cpp)
target_include_directories(oneapi_profile_pagani_kernel_f_6D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_pagani_kernel_f_6D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_pagani_kernel_f_6D PRIVATE "-lonemkl")

add_executable(oneapi_profile_pagani_kernel_f_8D oneapi_profile_pagani_kernel_f_8D.cpp)
target_include_directories(oneapi_profile_pagani_kernel_f_8D PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_profile_pagani_kernel_f_8D PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_profile_pagani_kernel_f_8D PRIVATE "-lonemkl")

add_executable(oneapi_execute_Addition_integrands_on_device execute_Addition_integrands_on_device.cpp)
target_include_directories(oneapi_execute_Addition_integrands_on_device PRIVATE "${ONEMKL_DIR}/include")
target_link_directories(oneapi_execute_Addition_integrands_on_device PUBLIC "${ONEMKL_DIR}/lib/")
target_compile_options(oneapi_execute_Addition_integrands_on_device PRIVATE "-lonemkl")