#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(new_interface_Genz4_6D new_interface_Genz4_6D.cu)
set_target_properties(new_interface_Genz4_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(new_interface_Genz4_6D util)
target_compile_options(new_interface_Genz4_6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz4_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
 
add_executable(new_interface_SinSum6D new_interface_SinSum_6D.cu)
set_target_properties(new_interface_SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(new_interface_SinSum6D util )
target_compile_options(new_interface_SinSum6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_SinSum6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz1_8D new_interface_Genz1_8D.cu)
set_target_properties(new_interface_Genz1_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(new_interface_Genz1_8D util )
target_compile_options(new_interface_Genz1_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz1_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(custom_funcs custom_funcs.cu)
set_target_properties(custom_funcs PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(custom_funcs util )
target_compile_options(custom_funcs PRIVATE "--expt-relaxed-constexpr")
target_include_directories(custom_funcs PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(hybrid hybrid.cu)
set_target_properties(hybrid PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(hybrid util )
target_compile_options(hybrid PRIVATE "--expt-relaxed-constexpr")
target_include_directories(hybrid PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz5_8D new_interface_Genz5_8D.cu)
set_target_properties(new_interface_Genz5_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(new_interface_Genz5_8D util )
target_compile_options(new_interface_Genz5_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz5_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz6_6D new_interface_Genz6_6D.cu)
set_target_properties(new_interface_Genz6_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(new_interface_Genz6_6D util )
target_compile_options(new_interface_Genz6_6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz6_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(new_interface_Genz2_6D new_interface_Genz2_6D.cu)
set_target_properties(new_interface_Genz2_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(new_interface_Genz2_6D util )
target_compile_options(new_interface_Genz2_6D PRIVATE "--expt-relaxed-constexpr" "-g" "-lineinfo")
target_include_directories(new_interface_Genz2_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz3_3D new_interface_Genz3_3D.cu)
set_target_properties(new_interface_Genz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(new_interface_Genz3_3D util )
target_compile_options(new_interface_Genz3_3D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz3_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz4_5D new_interface_Genz4_5D.cu)
set_target_properties(new_interface_Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(new_interface_Genz4_5D util )
target_compile_options(new_interface_Genz4_5D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz4_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(randomized_integrands randomized_integrands.cu)
set_target_properties(randomized_integrands PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(randomized_integrands util )
target_compile_options(randomized_integrands PRIVATE "--expt-relaxed-constexpr")
target_include_directories(randomized_integrands PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(cudaCuhre cudaCuhre.cu)
set_target_properties(cudaCuhre PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(cudaCuhre )
target_compile_options(cudaCuhre PRIVATE "--expt-relaxed-constexpr")
target_include_directories(cudaCuhre PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_B8_15 B8_15.cu)
set_target_properties(pagani_B8_15 PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_B8_15 util )
target_compile_options(pagani_B8_15 PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_B8_15 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_DiagonalRidge2D diagonalRidge2D.cu)
set_target_properties(pagani_DiagonalRidge2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_DiagonalRidge2D util )
target_compile_options(pagani_DiagonalRidge2D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_DiagonalRidge2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_B8_22 B8_22.cu)
set_target_properties(pagani_B8_22 PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_B8_22 util )
target_compile_options(pagani_B8_22 PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_B8_22 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Gauss9D Gauss9D.cu)
set_target_properties(pagani_Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Gauss9D util )
target_compile_options(pagani_Gauss9D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Gauss9D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_absCosSum5Dplus1 absCosSum5Dplus1.cu)
set_target_properties(pagani_absCosSum5Dplus1 PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_absCosSum5Dplus1 util )
target_compile_options(pagani_absCosSum5Dplus1 PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_absCosSum5Dplus1 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(pagani_Genz2_2D Genz2_2D.cu)
set_target_properties(pagani_Genz2_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz2_2D util )
target_compile_options(pagani_Genz2_2D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz2_2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz2_6D Genz2_6D.cu)
set_target_properties(pagani_Genz2_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz2_6D util )
target_compile_options(pagani_Genz2_6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz2_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz2_8D Genz2_8D.cu)
set_target_properties(pagani_Genz2_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz2_8D util )
target_compile_options(pagani_Genz2_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz2_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz3_3D Genz3_3D.cu)
set_target_properties(pagani_Genz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz3_3D util )
target_compile_options(pagani_Genz3_3D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz3_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz3_6D Genz3_6D.cu)
set_target_properties(pagani_Genz3_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz3_6D util )
target_compile_options(pagani_Genz3_6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz3_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz3_8D Genz3_8D.cu)
set_target_properties(pagani_Genz3_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz3_8D util )
target_compile_options(pagani_Genz3_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz3_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz4_5D Genz4_5D.cu)
set_target_properties(pagani_Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz4_5D util ) 
target_compile_options(pagani_Genz4_5D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz4_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz4_8D Genz4_8D.cu)
set_target_properties(pagani_Genz4_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz4_8D util )
target_compile_options(pagani_Genz4_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz4_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz5_2D Genz5_2D.cu)
set_target_properties(pagani_Genz5_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz5_2D util )
target_compile_options(pagani_Genz5_2D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz5_2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz5_5D Genz5_5D.cu)
set_target_properties(pagani_Genz5_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz5_5D util )
target_compile_options(pagani_Genz5_5D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz5_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz5_8D Genz5_8D.cu)
set_target_properties(pagani_Genz5_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz5_8D util )
target_compile_options(pagani_Genz5_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz5_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(pagani_Genz6_2D Genz6_2D.cu)
set_target_properties(pagani_Genz6_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz6_2D util )
target_compile_options(pagani_Genz6_2D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz6_2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz6_6D Genz6_6D.cu)
set_target_properties(pagani_Genz6_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz6_6D util )
target_compile_options(pagani_Genz6_6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz6_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz1_8D Genz1_8D.cu)
target_compile_options(pagani_Genz1_8D PRIVATE "-lineinfo")
set_target_properties(pagani_Genz1_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz1_8D util )
target_compile_options(pagani_Genz1_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Genz1_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(pagani_absCosSum5D absCosSum5D.cu)
target_compile_options(pagani_absCosSum5D PRIVATE "-lineinfo")
set_target_properties(pagani_absCosSum5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_absCosSum5D util )
target_compile_options(pagani_absCosSum5D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_absCosSum5D PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(pagani_genz_1abs_5d genz_1abs_5d.cu)
set_target_properties(pagani_genz_1abs_5d PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_genz_1abs_5d util )
target_compile_options(pagani_genz_1abs_5d PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_genz_1abs_5d PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(pagani_SinSum6D SinSum6D.cu)
set_target_properties(pagani_SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_SinSum6D util )
target_compile_options(pagani_SinSum6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_SinSum6D PRIVATE
  ${CMAKE_SOURCE_DIR}
)																								
