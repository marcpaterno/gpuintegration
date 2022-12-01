#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_subdirectory(simple_funcs)

add_executable(execute_integrands_on_device execute_integrands_on_device.cu)
set_target_properties(execute_integrands_on_device PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(execute_integrands_on_device util)
target_compile_options(execute_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(execute_integrands_on_device PRIVATE
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
