add_library(util SHARED cudaUtil.cu)
set_property(TARGET util PROPERTY CUDA_ARCHITECTURES 70-real)

add_library(testmain SHARED main.cc)
target_include_directories(testmain PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/externals)

add_executable(test_Volume test_Volume.cu)
target_link_libraries(test_Volume PRIVATE util testmain)
target_compile_options(test_Volume PRIVATE "${QUAD_WARNINGS}")
set_property(TARGET test_Volume PROPERTY CUDA_ARCHITECTURES 70-real)
add_test(test_Volume test_Volume)

