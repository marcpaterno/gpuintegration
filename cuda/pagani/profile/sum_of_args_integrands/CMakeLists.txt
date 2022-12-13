#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )


add_executable(profile_pagani_kernel_f_3D profile_pagani_kernel_f_3D.cu)
set_target_properties(profile_pagani_kernel_f_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_kernel_f_3D util)
target_compile_options(profile_pagani_kernel_f_3D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_kernel_f_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(profile_pagani_kernel_f_5D profile_pagani_kernel_f_5D.cu)
set_target_properties(profile_pagani_kernel_f_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_kernel_f_5D util)
target_compile_options(profile_pagani_kernel_f_5D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_kernel_f_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)



add_executable(profile_pagani_kernel_f_6D profile_pagani_kernel_f_6D.cu)
set_target_properties(profile_pagani_kernel_f_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_kernel_f_6D util)
target_compile_options(profile_pagani_kernel_f_6D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_kernel_f_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(profile_pagani_kernel_f_8D profile_pagani_kernel_f_8D.cu)
set_target_properties(profile_pagani_kernel_f_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(profile_pagani_kernel_f_8D util)
target_compile_options(profile_pagani_kernel_f_8D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_kernel_f_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

