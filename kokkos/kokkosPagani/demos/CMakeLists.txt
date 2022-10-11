#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(kokkos_pagani_Genz1_8D Genz1_8D.cpp)
target_compile_options(kokkos_pagani_Genz1_8D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz1_8D PUBLIC Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_Genz2_2D Genz2_2D.cpp)
target_compile_options(kokkos_pagani_Genz2_2D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz2_2D Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_Genz3_3D Genz3_3D.cpp)
target_compile_options(kokkos_pagani_Genz3_3D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz3_3D Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_Genz3_8D Genz3_8D.cpp)
target_compile_options(kokkos_pagani_Genz3_8D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz3_8D Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_Genz4_5D Genz4_5D.cpp)
target_compile_options(kokkos_pagani_Genz4_5D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz4_5D Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_Genz4_8D Genz4_8D.cpp)
target_compile_options(kokkos_pagani_Genz4_8D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz4_8D Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_Genz5_5D Genz5_5D.cpp)
target_compile_options(kokkos_pagani_Genz5_5D PRIVATE  "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz5_5D Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_Genz5_8D Genz5_8D.cpp)
target_compile_options(kokkos_pagani_Genz5_8D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz5_8D Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_Genz6_2D Genz6_2D.cpp)
target_compile_options(kokkos_pagani_Genz6_2D PRIVATE  "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz6_2D Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_Genz6_6D Genz6_6D.cpp)
target_compile_options(kokkos_pagani_Genz6_6D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Genz6_6D Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_B8_15 B8_15.cpp)
target_compile_options(kokkos_pagani_B8_15 PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_B8_15 Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})

add_executable(kokkos_pagani_B8_22 B8_22.cpp)
target_compile_options(kokkos_pagani_B8_22 PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_B8_22 Kokkos::kokkos Kokkos::kokkoskernels ${NVTX_LIBRARY})