add_executable(vegasT vegasT.cu)
target_compile_options(vegasT PRIVATE "-arch=sm_70")
set_target_properties(vegasT PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(vegasT PRIVATE ${CMAKE_SOURCE_DIR})
