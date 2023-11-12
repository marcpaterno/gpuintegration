add_executable(Genz3_8D Genz3_8D.cpp)
target_compile_options(Genz3_8D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(Genz3_8D Kokkos::kokkos)
target_include_directories(Genz3_8D PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(kokkos_mcubes_genz_integrals genz_integrals.cpp)
target_compile_options(kokkos_mcubes_genz_integrals PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_mcubes_genz_integrals Kokkos::kokkos)
target_include_directories(kokkos_mcubes_genz_integrals PRIVATE ${CMAKE_SOURCE_DIR})