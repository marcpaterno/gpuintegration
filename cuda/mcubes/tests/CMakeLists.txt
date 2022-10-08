add_executable(kernel_reduction kernel_reduction.cu)
target_compile_options(kernel_reduction PRIVATE "-DCURAND")
set_target_properties(kernel_reduction PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_include_directories(kernel_reduction PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(kernel_reduction kernel_reduction)
