#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(cuda_pagani_gaussians gaussians.cu)
set_target_properties(cuda_pagani_gaussians PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(cuda_pagani_gaussians util)
target_compile_options(cuda_pagani_gaussians PRIVATE "--expt-relaxed-constexpr")
target_include_directories(cuda_pagani_gaussians PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(cuda_pagani_discontinuous discontinuous.cu)
set_target_properties(cuda_pagani_discontinuous PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(cuda_pagani_discontinuous util)
target_compile_options(cuda_pagani_discontinuous PRIVATE "--expt-relaxed-constexpr")
target_include_directories(cuda_pagani_discontinuous PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(cuda_pagani_product_peaks product_peaks.cu)
set_target_properties(cuda_pagani_product_peaks PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(cuda_pagani_product_peaks util)
target_compile_options(cuda_pagani_product_peaks PRIVATE "--expt-relaxed-constexpr")
target_include_directories(cuda_pagani_product_peaks PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(cuda_pagani_c_zero_funcs c_zero_funcs.cu)
set_target_properties(cuda_pagani_c_zero_funcs PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(cuda_pagani_c_zero_funcs util)
target_compile_options(cuda_pagani_c_zero_funcs PRIVATE "--expt-relaxed-constexpr")
target_include_directories(cuda_pagani_c_zero_funcs PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(cuda_pagani_g_func g_function.cu)
set_target_properties(cuda_pagani_g_func PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(cuda_pagani_g_func util)
target_compile_options(cuda_pagani_g_func PRIVATE "--expt-relaxed-constexpr")
target_include_directories(cuda_pagani_g_func PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(oscillatory_integrands oscillatory_integrands.cu)
set_target_properties(oscillatory_integrands PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(oscillatory_integrands util)
target_compile_options(oscillatory_integrands PRIVATE "--expt-relaxed-constexpr")
target_include_directories(oscillatory_integrands PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(genz_integrals genz_integrals.cu)
set_target_properties(genz_integrals PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(genz_integrals util)
target_compile_options(genz_integrals PRIVATE "--expt-relaxed-constexpr")
target_include_directories(genz_integrals PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(addition_integrals addition_integrals.cu)
set_target_properties(addition_integrals PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(addition_integrals util)
target_compile_options(addition_integrals PRIVATE "--expt-relaxed-constexpr")
target_include_directories(addition_integrals PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz4_6D new_interface_Genz4_6D.cu)
set_target_properties(new_interface_Genz4_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(new_interface_Genz4_6D util)
target_compile_options(new_interface_Genz4_6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz4_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
 
add_executable(new_interface_SinSum6D new_interface_SinSum_6D.cu)
set_target_properties(new_interface_SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(new_interface_SinSum6D util )
target_compile_options(new_interface_SinSum6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_SinSum6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz1_8D new_interface_Genz1_8D.cu)
set_target_properties(new_interface_Genz1_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(new_interface_Genz1_8D util )
target_compile_options(new_interface_Genz1_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz1_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(custom_funcs custom_funcs.cu)
set_target_properties(custom_funcs PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(custom_funcs util )
target_compile_options(custom_funcs PRIVATE "--expt-relaxed-constexpr")
target_include_directories(custom_funcs PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(hybrid hybrid.cu)
set_target_properties(hybrid PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(hybrid util )
target_compile_options(hybrid PRIVATE "--expt-relaxed-constexpr")
target_include_directories(hybrid PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz5_8D new_interface_Genz5_8D.cu)
set_target_properties(new_interface_Genz5_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(new_interface_Genz5_8D util )
target_compile_options(new_interface_Genz5_8D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz5_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz6_6D new_interface_Genz6_6D.cu)
set_target_properties(new_interface_Genz6_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(new_interface_Genz6_6D util )
target_compile_options(new_interface_Genz6_6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz6_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(new_interface_Genz2_6D new_interface_Genz2_6D.cu)
set_target_properties(new_interface_Genz2_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(new_interface_Genz2_6D util )
target_compile_options(new_interface_Genz2_6D PRIVATE "--expt-relaxed-constexpr" "-g" "-lineinfo")
target_include_directories(new_interface_Genz2_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz3_3D new_interface_Genz3_3D.cu)
set_target_properties(new_interface_Genz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(new_interface_Genz3_3D util )
target_compile_options(new_interface_Genz3_3D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz3_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(new_interface_Genz4_5D new_interface_Genz4_5D.cu)
set_target_properties(new_interface_Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(new_interface_Genz4_5D util )
target_compile_options(new_interface_Genz4_5D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(new_interface_Genz4_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(randomized_integrands randomized_integrands.cu)
set_target_properties(randomized_integrands PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(randomized_integrands util )
target_compile_options(randomized_integrands PRIVATE "--expt-relaxed-constexpr")
target_include_directories(randomized_integrands PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(cudaCuhre cudaCuhre.cu)
set_target_properties(cudaCuhre PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(cudaCuhre )
target_compile_options(cudaCuhre PRIVATE "--expt-relaxed-constexpr")
target_include_directories(cudaCuhre PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_B8_15 B8_15.cu)
set_target_properties(pagani_B8_15 PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(pagani_B8_15 util )
target_compile_options(pagani_B8_15 PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_B8_15 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_DiagonalRidge2D diagonalRidge2D.cu)
set_target_properties(pagani_DiagonalRidge2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(pagani_DiagonalRidge2D util )
target_compile_options(pagani_DiagonalRidge2D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_DiagonalRidge2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_B8_22 B8_22.cu)
set_target_properties(pagani_B8_22 PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(pagani_B8_22 util )
target_compile_options(pagani_B8_22 PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_B8_22 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Gauss9D Gauss9D.cu)
set_target_properties(pagani_Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(pagani_Gauss9D util )
target_compile_options(pagani_Gauss9D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_Gauss9D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_absCosSum5Dplus1 absCosSum5Dplus1.cu)
set_target_properties(pagani_absCosSum5Dplus1 PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(pagani_absCosSum5Dplus1 util )
target_compile_options(pagani_absCosSum5Dplus1 PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_absCosSum5Dplus1 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_absCosSum5D absCosSum5D.cu)
target_compile_options(pagani_absCosSum5D PRIVATE "-lineinfo")
set_target_properties(pagani_absCosSum5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(pagani_absCosSum5D util )
target_compile_options(pagani_absCosSum5D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_absCosSum5D PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(pagani_genz_1abs_5d genz_1abs_5d.cu)
set_target_properties(pagani_genz_1abs_5d PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(pagani_genz_1abs_5d util )
target_compile_options(pagani_genz_1abs_5d PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_genz_1abs_5d PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(pagani_SinSum6D SinSum6D.cu)
set_target_properties(pagani_SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(pagani_SinSum6D util )
target_compile_options(pagani_SinSum6D PRIVATE "--expt-relaxed-constexpr")
target_include_directories(pagani_SinSum6D PRIVATE
  ${CMAKE_SOURCE_DIR}
)																								
