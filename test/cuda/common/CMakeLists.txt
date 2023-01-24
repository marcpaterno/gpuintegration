SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-math-errno -mtune=native" )

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Turn on compiler warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT DEFINED TARGET_ARCH)
	MESSAGE(FATAL_ERROR "You must specify -DPAGANI_TARGET_ARCH=<arch> to generate code for the right GPU, e.g. '=70-real' for V100")
endif()

include_directories(AFTER ${CMAKE_SOURCE_DIR} ${TOP_DIR} externals)

set(CMAKE_SHARED_MODULE_PREFIX "")    # We don't need the leading lib

#find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH )

add_executable(min_max min_max.cu)
set_target_properties(min_max PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_include_directories(min_max PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
target_compile_options(min_max PRIVATE "--expt-relaxed-constexpr")
add_test(min_max min_max)

add_executable(prefix_sum prefix_sum.cu)
set_target_properties(prefix_sum PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_include_directories(prefix_sum PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(prefix_sum prefix_sum)

add_executable(cudaDynamicArray cudaDynamicArray.cu)
target_link_libraries(cudaDynamicArray cpu_model gpu_model)
set_target_properties(cudaDynamicArray PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_include_directories(cudaDynamicArray PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(cudaDynamicArray cudaDynamicArray)


add_executable(hello hello.cu)
set_target_properties(hello PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
add_test(hello hello)

add_executable(StatefullIntegrand StatefulIntegrand.cu)
set_target_properties(StatefullIntegrand PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(StatefullIntegrand )
target_include_directories(StatefullIntegrand PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(StatefullIntegrand StatefullIntegrand)

add_executable(test_volume test_volume.cu)
set_target_properties(test_volume PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(test_volume util)
target_include_directories(test_volume PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_volume test_volume)


add_executable(Interpolation1D Interpolation1D.cu)
set_target_properties(Interpolation1D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(Interpolation1D )
target_include_directories(Interpolation1D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Interpolation1D Interpolation1D)

add_executable(Interpolation2D Interpolation2D.cu)
set_target_properties(Interpolation2D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(Interpolation2D )
target_include_directories(Interpolation2D PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Interpolation2D Interpolation2D)

add_executable(interpolator_mem_usage interpolator_mem_usage.cu)
set_target_properties(interpolator_mem_usage PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_compile_options(interpolator_mem_usage PRIVATE "-g" "-lineinfo") 
target_link_libraries(interpolator_mem_usage)
target_include_directories(interpolator_mem_usage PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(Reduction Reduction.cu)
set_target_properties(Reduction PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_compile_options(Reduction PRIVATE "-g" "-lineinfo") 
target_link_libraries(Reduction)
target_include_directories(Reduction PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(Reduction Reduction)
