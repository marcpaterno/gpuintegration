add_executable(kokkos_pagani_GenerateRegions GenerateInitialRegions.cpp)
target_compile_options(kokkos_pagani_GenerateRegions PRIVATE "-O2" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_GenerateRegions Kokkos::kokkos Kokkos::kokkoskernels)
add_test(kokkos_pagani_GenerateRegions kokkos_pagani_GenerateRegions)

add_executable(kokkos_pagani_RegionSampling RegionSampling.cpp)
target_compile_options(kokkos_pagani_RegionSampling PRIVATE "-O2" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_RegionSampling Kokkos::kokkos Kokkos::kokkoskernels)
add_test(kokkos_pagani_RegionSampling kokkos_pagani_RegionSampling)

add_executable(kokkos_pagani_MemoryUsage MemoryUsage.cpp)
target_compile_options(kokkos_pagani_MemoryUsage PRIVATE "-O2" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_MemoryUsage Kokkos::kokkos Kokkos::kokkoskernels)
add_test(kokkos_pagani_MemoryUsage kokkos_pagani_MemoryUsage)

add_executable(kokkos_pagani_Interpolation1D Interpolation1D.cpp)
target_compile_options(kokkos_pagani_Interpolation1D PRIVATE "-O2" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Interpolation1D Kokkos::kokkos Kokkos::kokkoskernels)
add_test(kokkos_pagani_Interpolation1D kokkos_pagani_Interpolation1D)

add_executable(kokkos_pagani_Interpolation2D Interpolation2D.cpp)
target_compile_options(kokkos_pagani_Interpolation2D PRIVATE "-O2" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_Interpolation2D Kokkos::kokkos Kokkos::kokkoskernels)
add_test(kokkos_pagani_Interpolation2D kokkos_pagani_Interpolation2D)