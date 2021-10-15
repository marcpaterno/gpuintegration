add_executable(kernel_reduction kernel_reduction.cu)
target_compile_options(kernel_reduction PRIVATE "-arch=sm_70" "-O2" "-DCURAND")
set_target_properties(kernel_reduction PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kernel_reduction PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)