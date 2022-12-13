#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_subdirectory(simple_funcs)
add_subdirectory(sum_of_args_integrands)

add_executable(profile_integrands profile_integrands.cu)
set_target_properties(profile_integrands PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_integrands util)
target_compile_options(profile_integrands PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_integrands PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(execute_math_functions_on_device execute_math_functions_on_device.cu)
set_target_properties(execute_math_functions_on_device PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(execute_math_functions_on_device util)
target_compile_options(execute_math_functions_on_device PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(execute_math_functions_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(execute_8D_benchmark_integrands_on_device execute_8D_benchmark_integrands_on_device.cu)
set_target_properties(execute_8D_benchmark_integrands_on_device PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(execute_8D_benchmark_integrands_on_device util)
target_compile_options(execute_8D_benchmark_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(execute_8D_benchmark_integrands_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(execute_7D_benchmark_integrands_on_device execute_7D_benchmark_integrands_on_device.cu)
set_target_properties(execute_7D_benchmark_integrands_on_device PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(execute_7D_benchmark_integrands_on_device util)
target_compile_options(execute_7D_benchmark_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(execute_7D_benchmark_integrands_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(execute_6D_benchmark_integrands_on_device execute_6D_benchmark_integrands_on_device.cu)
set_target_properties(execute_6D_benchmark_integrands_on_device PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(execute_6D_benchmark_integrands_on_device util)
target_compile_options(execute_6D_benchmark_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(execute_6D_benchmark_integrands_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(execute_5D_benchmark_integrands_on_device execute_5D_benchmark_integrands_on_device.cu)
set_target_properties(execute_5D_benchmark_integrands_on_device PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(execute_5D_benchmark_integrands_on_device util)
target_compile_options(execute_5D_benchmark_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(execute_5D_benchmark_integrands_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(profile_pagani_Genz2_6D profile_pagani_Genz2_6D.cu)
set_target_properties(profile_pagani_Genz2_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_Genz2_6D util)
target_compile_options(profile_pagani_Genz2_6D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_Genz2_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
 
add_executable(profile_pagani_Genz3_3D profile_pagani_Genz3_3D.cu)
set_target_properties(profile_pagani_Genz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_Genz3_3D util)
target_compile_options(profile_pagani_Genz3_3D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_Genz3_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(profile_pagani_Genz3_8D profile_pagani_Genz3_8D.cu)
set_target_properties(profile_pagani_Genz3_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_Genz3_8D util)
target_compile_options(profile_pagani_Genz3_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(profile_pagani_Genz3_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(profile_pagani_Genz4_5D profile_pagani_Genz4_5D.cu)
set_target_properties(profile_pagani_Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_Genz4_5D util)
target_compile_options(profile_pagani_Genz4_5D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_Genz4_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(profile_pagani_Genz5_8D profile_pagani_Genz5_8D.cu)
set_target_properties(profile_pagani_Genz5_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_Genz5_8D util)
target_compile_options(profile_pagani_Genz5_8D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_Genz5_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(profile_pagani_Genz6_6D profile_pagani_Genz6_6D.cu)
set_target_properties(profile_pagani_Genz6_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_Genz6_6D util)
target_compile_options(profile_pagani_Genz6_6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(profile_pagani_Genz6_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
