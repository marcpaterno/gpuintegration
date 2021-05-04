add_executable(Genz1_8D Genz1_8D.cpp)
target_compile_options(Genz1_8D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz1_8D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Genz2_2D Genz2_2D.cpp)
target_compile_options(Genz2_2D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz2_2D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Genz3_3D Genz3_3D.cpp)
target_compile_options(Genz3_3D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz3_3D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Genz3_8D Genz3_8D.cpp)
target_compile_options(Genz3_8D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz3_8D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Genz4_5D Genz4_5D.cpp)
target_compile_options(Genz4_5D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz4_5D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Genz4_8D Genz4_8D.cpp)
target_compile_options(Genz4_8D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz4_8D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Genz5_5D Genz5_5D.cpp)
target_compile_options(Genz5_5D PRIVATE  "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz5_5D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Genz5_8D Genz5_8D.cpp)
target_compile_options(Genz5_8D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz5_8D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Genz6_2D Genz6_2D.cpp)
target_compile_options(Genz6_2D PRIVATE  "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz6_2D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Genz6_6D Genz6_6D.cpp)
target_compile_options(Genz6_6D PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Genz6_6D Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(B8_15 B8_15.cpp)
target_compile_options(B8_15 PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(B8_15 Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(B8_22 B8_22.cpp)
target_compile_options(B8_22 PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(B8_22 Kokkos::kokkos Kokkos::kokkoskernels)

add_executable(Init_Region_Pool Init_Region_Pool.cpp)
target_compile_options(Init_Region_Pool PRIVATE "-arch=sm_70" "--expt-relaxed-constexpr")
target_link_libraries(Init_Region_Pool Kokkos::kokkos Kokkos::kokkoskernels)