#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(kokkos_profile_pagani_3D profile_pagani_3D.cpp)
target_compile_options(kokkos_profile_pagani_3D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_3D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(kokkos_profile_pagani_5D profile_pagani_5D.cpp)
target_compile_options(kokkos_profile_pagani_5D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_5D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(kokkos_profile_pagani_6D profile_pagani_6D.cpp)
target_compile_options(kokkos_profile_pagani_6D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_6D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
 

add_executable(kokkos_profile_pagani_8D profile_pagani_8D.cpp)
target_compile_options(kokkos_profile_pagani_8D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_profile_pagani_8D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

