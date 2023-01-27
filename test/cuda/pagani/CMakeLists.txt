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

add_executable(easy Easy_Integrals.cu)
set_target_properties(easy PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_include_directories(easy PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
target_compile_options(easy PRIVATE "--expt-relaxed-constexpr")
add_test(easy easy)

add_executable(cuda_uniform_sub_division Uniform_sub_division.cu)
set_target_properties(cuda_uniform_sub_division PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_executable(pagani_loop_test pagani_loop_test.cu)
set_target_properties(pagani_loop_test PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})

add_library(cpu_model model.cc)
target_include_directories(cpu_model
                           PRIVATE ${CMAKE_SOURCE_DIR}
                                   ${CMAKE_SOURCE_DIR}/externals)

add_library(gpu_model model.cu)
set_target_properties(gpu_model PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(gpu_model )
target_include_directories(gpu_model PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(test_cpu_vs_gpu test_cpu_vs_gpu.cu)
target_link_libraries(test_cpu_vs_gpu cpu_model gpu_model)
set_target_properties(test_cpu_vs_gpu PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_include_directories(test_cpu_vs_gpu PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(test_cpu_vs_gpu test_cpu_vs_gpu)

add_library(do_integration_from_c do_integration_from_c.cu)
set_target_properties(do_integration_from_c PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(do_integration_from_c util )
target_include_directories(do_integration_from_c PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)

add_executable(test_integration_from_c test_integration_from_c.c)
target_link_libraries(test_integration_from_c do_integration_from_c)
add_test(test_integration_from_c test_integration_from_c)

add_executable(accuracy_improves_with_epsrel accuracy_improves_with_epsrel.cu)
set_target_properties(accuracy_improves_with_epsrel PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(accuracy_improves_with_epsrel util )
target_include_directories(accuracy_improves_with_epsrel PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(accuracy_improves_with_epsrel accuracy_improves_with_epsrel)

add_executable(BoxIntegrals BoxIntegrals.cu)
set_target_properties(BoxIntegrals PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(BoxIntegrals util )
target_include_directories(BoxIntegrals PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(BoxIntegrals BoxIntegrals)

add_executable(NanIntegral NanIntegral.cu)
set_target_properties(NanIntegral PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(NanIntegral util )
target_include_directories(NanIntegral PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(NanIntegral NanIntegral)

add_executable(RegionSampling RegionSampling.cu)
set_target_properties(RegionSampling PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${TARGET_ARCH})
target_link_libraries(RegionSampling util )
target_include_directories(RegionSampling PRIVATE
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/externals
)
add_test(RegionSampling RegionSampling)