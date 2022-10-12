add_executable(Genz3_8D Genz3_8D.cpp)
target_compile_options(Genz3_8D PRIVATE "--expt-relaxed-constexpr")
target_link_libraries(Genz3_8D Kokkos::kokkos)
target_include_directories(Genz3_8D PRIVATE ${CMAKE_SOURCE_DIR})
