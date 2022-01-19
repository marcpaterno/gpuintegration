#cmake_minimum_required (VERSION 3.10)
#project (kokkosPagani)
#enable_testing()

set(Kokkos_DIR "$ENV{Kokkos_ROOT}" CACHE STRING "Kokkos root directory")

#add_subdirectory(kokkos)
#add_subdirectory(kokkos-kernels)
find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
include_directories(quad)
#include_directories(externals)

add_subdirectory(demos)
add_subdirectory(tests)
#add_executable(pagani main.cpp)
#target_compile_options(pagani PRIVATE "-O2" "-DCUSTOM" "-arch=sm_70" "--expt-relaxed-constexpr" "-std=c++17")
#target_link_libraries(pagani Kokkos::kokkos Kokkos::kokkoskernels)

