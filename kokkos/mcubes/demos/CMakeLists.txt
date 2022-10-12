add_executable(Genz3_8D Genz3_8D.cpp)
target_compile_options(Genz3_8D PRIVATE "-O2" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz3_8D Kokkos::kokkos)

