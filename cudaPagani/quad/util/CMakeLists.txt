add_library(quadutil SHARED cudaUtil.cu)
set_property(TARGET quadutil PROPERTY CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})

add_executable(test_Volume test_Volume.cu)
target_link_libraries(test_Volume PRIVATE quadutil)
target_compile_options(test_Volume PRIVATE "${QUAD_WARNINGS}")
set_property(TARGET test_Volume PROPERTY CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
add_test(test_Volume test_Volume)

