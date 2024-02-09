add_executable(kokkos_mcubes_Genz3_8D Genz3_8D.cpp)
target_compile_options(kokkos_mcubes_Genz3_8D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_mcubes_Genz3_8D Kokkos::kokkos)
target_include_directories(kokkos_mcubes_Genz3_8D PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(kokkos_mcubes_Genz4_5D Genz4_5D.cpp)
target_compile_options(kokkos_mcubes_Genz4_5D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_mcubes_Genz4_5D Kokkos::kokkos)
target_include_directories(kokkos_mcubes_Genz4_5D PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(kokkos_mcubes_Genz5_8D Genz5_8D.cpp)
target_compile_options(kokkos_mcubes_Genz5_8D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_mcubes_Genz5_8D Kokkos::kokkos)
target_include_directories(kokkos_mcubes_Genz5_8D PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(kokkos_mcubes_Gauss9D Gauss9D.cpp)
target_compile_options(kokkos_mcubes_Gauss9D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_mcubes_Gauss9D Kokkos::kokkos)
target_include_directories(kokkos_mcubes_Gauss9D PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(kokkos_mcubes_SinSum6D SinSum6D.cpp)
target_compile_options(kokkos_mcubes_SinSum6D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_mcubes_SinSum6D Kokkos::kokkos)
target_include_directories(kokkos_mcubes_SinSum6D PRIVATE ${CMAKE_SOURCE_DIR})

add_executable(kokkos_mcubes_genz_integrals genz_integrals.cpp)
target_compile_options(kokkos_mcubes_genz_integrals PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(kokkos_mcubes_genz_integrals Kokkos::kokkos)
target_include_directories(kokkos_mcubes_genz_integrals PRIVATE ${CMAKE_SOURCE_DIR})
