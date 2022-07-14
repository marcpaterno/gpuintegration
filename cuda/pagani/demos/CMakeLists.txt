find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(randomized_integrands randomized_integrands.cu)
set_target_properties(randomized_integrands PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(randomized_integrands util ${NVTX_LIBRARY})
target_include_directories(randomized_integrands PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(cudaCuhre cudaCuhre.cu)
set_target_properties(cudaCuhre PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(cudaCuhre ${NVTX_LIBRARY})
target_include_directories(cudaCuhre PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_B8_15 B8_15.cu)
set_target_properties(pagani_B8_15 PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH} CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_B8_15 util ${NVTX_LIBRARY})
target_include_directories(pagani_B8_15 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_DiagonalRidge2D diagonalRidge2D.cu)
set_target_properties(pagani_DiagonalRidge2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_DiagonalRidge2D util ${NVTX_LIBRARY})
target_include_directories(pagani_DiagonalRidge2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_B8_22 B8_22.cu)
set_target_properties(pagani_B8_22 PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_B8_22 util ${NVTX_LIBRARY})
target_include_directories(pagani_B8_22 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Gauss9D Gauss9D.cu)
set_target_properties(pagani_Gauss9D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Gauss9D util ${NVTX_LIBRARY})
target_include_directories(pagani_Gauss9D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Gauss9Dfloat Gauss9Dfloat.cu)
set_target_properties(pagani_Gauss9Dfloat PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Gauss9Dfloat util ${NVTX_LIBRARY})
target_include_directories(pagani_Gauss9Dfloat PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_absCosSum5Dplus1 absCosSum5Dplus1.cu)
set_target_properties(pagani_absCosSum5Dplus1 PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_absCosSum5Dplus1 util ${NVTX_LIBRARY})
target_include_directories(pagani_absCosSum5Dplus1 PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(pagani_Genz2_2D Genz2_2D.cu)
set_target_properties(pagani_Genz2_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz2_2D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz2_2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz2_6D Genz2_6D.cu)
set_target_properties(pagani_Genz2_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz2_6D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz2_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz2_8D Genz2_8D.cu)
set_target_properties(pagani_Genz2_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz2_8D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz2_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz3_3D Genz3_3D.cu)
set_target_properties(pagani_Genz3_3D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz3_3D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz3_3D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz3_6D Genz3_6D.cu)
set_target_properties(pagani_Genz3_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz3_6D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz3_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz3_8D Genz3_8D.cu)
set_target_properties(pagani_Genz3_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz3_8D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz3_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz4_5D Genz4_5D.cu)
set_target_properties(pagani_Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz4_5D util ${NVTX_LIBRARY}) 
target_include_directories(pagani_Genz4_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz4_5Dfloat Genz4_5Dfloat.cu)
set_target_properties(pagani_Genz4_5Dfloat PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz4_5Dfloat util ${NVTX_LIBRARY}) 
target_include_directories(pagani_Genz4_5Dfloat PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz4_8D Genz4_8D.cu)
set_target_properties(pagani_Genz4_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz4_8D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz4_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz4_8Dfloat Genz4_8Dfloat.cu)
set_target_properties(pagani_Genz4_8Dfloat PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz4_8Dfloat util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz4_8Dfloat PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz5_2D Genz5_2D.cu)
set_target_properties(pagani_Genz5_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz5_2D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz5_2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz5_5D Genz5_5D.cu)
set_target_properties(pagani_Genz5_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz5_5D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz5_5D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz5_8D Genz5_8D.cu)
set_target_properties(pagani_Genz5_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz5_8D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz5_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)


add_executable(pagani_Genz6_2D Genz6_2D.cu)
set_target_properties(pagani_Genz6_2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz6_2D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz6_2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz6_6D Genz6_6D.cu)
set_target_properties(pagani_Genz6_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz6_6D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz6_6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz6_6Dfloat Genz6_6Dfloat.cu)
set_target_properties(pagani_Genz6_6Dfloat PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz6_6Dfloat util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz6_6Dfloat PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(pagani_Genz1_8D Genz1_8D.cu)
target_compile_options(pagani_Genz1_8D PRIVATE "-lineinfo")
set_target_properties(pagani_Genz1_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz1_8D util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz1_8D PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(pagani_Genz1_8Dfloat Genz1_8Dfloat.cu)
target_compile_options(pagani_Genz1_8Dfloat PRIVATE "-lineinfo")
set_target_properties(pagani_Genz1_8Dfloat PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_Genz1_8Dfloat util ${NVTX_LIBRARY})
target_include_directories(pagani_Genz1_8Dfloat PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(pagani_absCosSum5D absCosSum5D.cu)
target_compile_options(pagani_absCosSum5D PRIVATE "-lineinfo")
set_target_properties(pagani_absCosSum5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_absCosSum5D util ${NVTX_LIBRARY})
target_include_directories(pagani_absCosSum5D PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(pagani_genz_1abs_5d genz_1abs_5d.cu)
set_target_properties(pagani_genz_1abs_5d PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_genz_1abs_5d util ${NVTX_LIBRARY})
target_include_directories(pagani_genz_1abs_5d PRIVATE
  ${CMAKE_SOURCE_DIR}
)

add_executable(pagani_SinSum6D SinSum6D.cu)
set_target_properties(pagani_SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(pagani_SinSum6D util ${NVTX_LIBRARY})
target_include_directories(pagani_SinSum6D PRIVATE
  ${CMAKE_SOURCE_DIR}
)																								