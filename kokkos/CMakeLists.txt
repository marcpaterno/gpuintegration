if (GPUINTEGRATION_BUILD_KOKKOS)
  find_package(Kokkos REQUIRED)
  find_package(KokkosKernels REQUIRED)
  add_subdirectory(kokkosPagani)
  add_subdirectory(mcubes)
endif()

