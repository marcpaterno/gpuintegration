add_library(util SHARED cudaUtil.cu)

add_library(testmain SHARED main.cc)
target_include_directories(testmain PUBLIC
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/externals)

add_executable(test_Volume test_Volume.cu)
target_link_libraries(test_Volume PRIVATE util testmain)
target_compile_options(test_Volume PRIVATE "${QUAD_WARNINGS}")
add_test(test_Volume test_Volume)

