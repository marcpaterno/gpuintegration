add_library(kokkos_catch_main STATIC kokkos_catch_main.cc)
target_link_libraries(kokkos_catch_main PRIVATE Kokkos::kokkos)
add_custom_command(TARGET kokkos_catch_main POST_BUILD
  COMMAND strip -S $<TARGET_FILE:kokkos_catch_main>
  )

add_executable(kokkos_pagani_GenerateRegions GenerateInitialRegions.cpp)
target_compile_options(kokkos_pagani_GenerateRegions PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_GenerateRegions Kokkos::kokkos Kokkos::kokkoskernels kokkos_catch_main)
target_include_directories(kokkos_pagani_GenerateRegions PRIVATE ${CMAKE_SOURCE_DIR})
add_test(kokkos_pagani_GenerateRegions kokkos_pagani_GenerateRegions)

add_executable(kokkos_pagani_RegionSampling RegionSampling.cpp)
target_compile_options(kokkos_pagani_RegionSampling PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_pagani_RegionSampling Kokkos::kokkos Kokkos::kokkoskernels kokkos_catch_main)
target_include_directories(kokkos_pagani_RegionSampling PRIVATE ${CMAKE_SOURCE_DIR})
add_test(kokkos_pagani_RegionSampling kokkos_pagani_RegionSampling)


add_executable(kokkos_RegionFiltering RegionFiltering.cpp)
target_compile_options(kokkos_RegionFiltering PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_RegionFiltering Kokkos::kokkos Kokkos::kokkoskernels kokkos_catch_main)
target_include_directories(kokkos_RegionFiltering PRIVATE ${CMAKE_SOURCE_DIR})
add_test(kokkos_RegionFiltering kokkos_RegionFiltering)
