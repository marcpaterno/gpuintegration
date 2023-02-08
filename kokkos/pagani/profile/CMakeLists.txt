#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_subdirectory(simple_funcs)
add_subdirectory(sum_of_args_integrands)


add_executable(kokkos_profile_integrands profile_integrands.cpp)
target_compile_options(kokkos_profile_integrands PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_integrands PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_integrands PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_profile_pagani_integrands profile_pagani_integrands.cpp)
target_compile_options(kokkos_profile_pagani_integrands PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_integrands PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_integrands PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_profile_mcubes_integrands profile_mcubes_integrands.cpp)
target_compile_options(kokkos_profile_mcubes_integrands PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_mcubes_integrands PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_mcubes_integrands PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_execute_math_functions_on_device execute_math_functions_on_device.cpp)
target_compile_options(kokkos_execute_math_functions_on_device PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_execute_math_functions_on_device PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_execute_math_functions_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_execute_8D_benchmark_integrands_on_device execute_8D_benchmark_integrands_on_device.cpp)
target_compile_options(kokkos_execute_8D_benchmark_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_execute_8D_benchmark_integrands_on_device PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_execute_8D_benchmark_integrands_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_execute_7D_benchmark_integrands_on_device execute_7D_benchmark_integrands_on_device.cpp)
target_compile_options(kokkos_execute_7D_benchmark_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_execute_7D_benchmark_integrands_on_device PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_execute_7D_benchmark_integrands_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_execute_6D_benchmark_integrands_on_device execute_6D_benchmark_integrands_on_device.cpp)
target_compile_options(kokkos_execute_6D_benchmark_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_execute_6D_benchmark_integrands_on_device PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_execute_6D_benchmark_integrands_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_execute_5D_benchmark_integrands_on_device execute_5D_benchmark_integrands_on_device.cpp)
target_compile_options(kokkos_execute_5D_benchmark_integrands_on_device PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_execute_5D_benchmark_integrands_on_device PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_execute_5D_benchmark_integrands_on_device PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(kokkos_profile_pagani_Genz2_6D profile_pagani_Genz2_6D.cpp)
target_compile_options(kokkos_profile_pagani_Genz2_6D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_Genz2_6D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_Genz2_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
 
add_executable(kokkos_profile_pagani_Genz3_3D profile_pagani_Genz3_3D.cpp)
target_compile_options(kokkos_profile_pagani_Genz3_3D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_Genz3_3D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_Genz3_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_profile_pagani_Genz3_8D profile_pagani_Genz3_8D.cpp)
target_compile_options(kokkos_profile_pagani_Genz3_8D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_profile_pagani_Genz3_8D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_Genz3_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_profile_pagani_Genz4_5D profile_pagani_Genz4_5D.cpp)
target_compile_options(kokkos_profile_pagani_Genz4_5D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_Genz4_5D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_Genz4_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_profile_pagani_Genz5_8D profile_pagani_Genz5_8D.cpp)
target_compile_options(kokkos_profile_pagani_Genz5_8D PRIVATE "--expt-relaxed-constexpr" )
target_link_libraries(kokkos_profile_pagani_Genz5_8D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_Genz5_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_profile_pagani_Genz6_6D profile_pagani_Genz6_6D.cpp)
target_compile_options(kokkos_profile_pagani_Genz6_6D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_profile_pagani_Genz6_6D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_profile_pagani_Genz6_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(kokkos_atomic_addition atomic_addition.cpp)
target_compile_options(kokkos_atomic_addition PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_atomic_addition PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})
target_include_directories(kokkos_atomic_addition PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)