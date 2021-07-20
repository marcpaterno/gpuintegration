add_executable(GenerateRegions GenerateInitialRegions.cpp)
target_compile_options(GenerateRegions PRIVATE "-O2" "-DCUSTOM" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(GenerateRegions Kokkos::kokkos Kokkos::kokkoskernels)
add_test(GenerateRegions GenerateRegions)

add_executable(RegionSampling RegionSampling.cpp)
target_compile_options(RegionSampling PRIVATE "-O2" "-DCUSTOM" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(RegionSampling Kokkos::kokkos Kokkos::kokkoskernels)
add_test(RegionSampling RegionSampling)

add_executable(MemoryUsage MemoryUsage.cpp)
target_compile_options(MemoryUsage PRIVATE "-O2" "-DCUSTOM" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(MemoryUsage Kokkos::kokkos Kokkos::kokkoskernels)
add_test(MemoryUsage MemoryUsage)

add_executable(Interpolation1D Interpolation1D.cpp)
target_compile_options(Interpolation1D PRIVATE "-O2" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Interpolation1D Kokkos::kokkos Kokkos::kokkoskernels)
add_test(Interpolation1D Interpolation1D)