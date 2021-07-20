cmake_minimum_required (VERSION 3.10)
project (kokkosPagani)
enable_testing()

set(Kokkos_DIR "$ENV{Kokkos_ROOT}" CACHE STRING "Kokkos root directory")

add_subdirectory(kokkos)
add_subdirectory(kokkos-kernels)
include_directories(quad)
include_directories(externals)

add_subdirectory(demos)

add_executable(pagani main.cpp)
target_compile_options(pagani PRIVATE "-O2" "-DCUSTOM" "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(pagani Kokkos::kokkos Kokkos::kokkoskernels)

add_subdirectory(tests)