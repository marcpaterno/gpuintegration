if (GPUINTEGRATION_BUILD_KOKKOS)
  find_package(Kokkos REQUIRED)
  find_package(KokkosKernels REQUIRED)
  add_subdirectory(pagani)
  add_subdirectory(mcubes)
endif()

