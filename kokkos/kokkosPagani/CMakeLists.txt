find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
include_directories(quad)

add_subdirectory(demos)
add_subdirectory(tests)

