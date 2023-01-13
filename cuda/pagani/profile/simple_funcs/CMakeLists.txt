#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(profile_pagani_3D profile_pagani_3D.cu)
set_target_properties(profile_pagani_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(profile_pagani_3D util)
target_compile_options(profile_pagani_3D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(profile_pagani_5D profile_pagani_5D.cu)
set_target_properties(profile_pagani_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(profile_pagani_5D util)
target_compile_options(profile_pagani_5D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(profile_pagani_6D profile_pagani_6D.cu)
set_target_properties(profile_pagani_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

target_link_libraries(profile_pagani_6D util)
target_compile_options(profile_pagani_6D PRIVATE "--expt-relaxed-constexpr" )
target_include_directories(profile_pagani_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
 

add_executable(profile_pagani_8D profile_pagani_8D.cu)
set_target_properties(profile_pagani_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(profile_pagani_8D util)
target_compile_options(profile_pagani_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(profile_pagani_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

