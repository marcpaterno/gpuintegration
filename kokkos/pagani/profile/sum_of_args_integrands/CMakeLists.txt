#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(kokkos_execute_Addition_integrands_on_device execute_Addition_integrands_on_device.cpp)
target_compile_options(kokkos_execute_Addition_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_execute_Addition_integrands_on_device PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_execute_Addition_integrands_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)



add_executable(kokkos_profile_pagani_kernel_f_3D profile_pagani_kernel_f_3D.cpp)
target_compile_options(kokkos_profile_pagani_kernel_f_3D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_kernel_f_3D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_kernel_f_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_profile_pagani_kernel_f_5D profile_pagani_kernel_f_5D.cpp)
target_compile_options(kokkos_profile_pagani_kernel_f_5D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_kernel_f_5D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_kernel_f_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)



add_executable(kokkos_profile_pagani_kernel_f_6D profile_pagani_kernel_f_6D.cpp)
target_compile_options(kokkos_profile_pagani_kernel_f_6D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_kernel_f_6D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_kernel_f_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(kokkos_profile_pagani_kernel_f_8D profile_pagani_kernel_f_8D.cpp)
target_compile_options(kokkos_profile_pagani_kernel_f_8D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_kernel_f_8D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_kernel_f_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

