add_executable(test_Volume test_Volume.cu)
target_link_libraries(test_Volume PRIVATE)
target_compile_options(test_Volume PRIVATE "${QUAD_WARNINGS}" "--expt-relaxed-constexpr")
set_property(TARGET test_Volume PROPERTY CUDA_ARCHITECTURES ${TARGET_ARCH})
add_test(test_Volume test_Volume)

