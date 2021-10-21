add_executable(vegasT vegasT.cu)
set_target_properties(vegasT PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_include_directories(vegasT PRIVATE ${CMAKE_SOURCE_DIR})
