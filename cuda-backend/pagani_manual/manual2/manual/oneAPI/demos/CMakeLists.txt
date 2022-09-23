add_executable(oneAPI_SinSum6D SinSum6D.cpp)
#set_target_properties(oneAPI_SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(oneAPI_SinSum6D PRIVATE MKL::MKL_DPCPP)
target_include_directories(oneAPI_SinSum6D PRIVATE
  ${CMAKE_SOURCE_DIR}
  )
  
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

add_executable(oneAPI_Genz3_3D Genz3_3D.cpp)
#set_target_properties(oneAPI_SinSum6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(oneAPI_Genz3_3D PRIVATE MKL::MKL_DPCPP)
target_include_directories(oneAPI_Genz3_3D PRIVATE
  ${CMAKE_SOURCE_DIR})
target_compile_options(oneAPI_Genz3_3D PRIVATE)

add_executable(oneAPI_Genz3_8D Genz3_8D.cpp)
#set_target_properties(oneAPI_Genz3_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(oneAPI_Genz3_8D PRIVATE MKL::MKL_DPCPP)
target_include_directories(oneAPI_Genz3_8D PRIVATE
  ${CMAKE_SOURCE_DIR})
target_compile_options(oneAPI_Genz3_8D PRIVATE)

add_executable(compute_generators compute_generators.cpp)
#set_target_properties(compute_generators PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(compute_generators PRIVATE MKL::MKL_DPCPP)
target_include_directories(compute_generators PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(compute_generators PRIVATE)

add_executable(hierarchical_parallelism hierarchical_parallelism.cpp)
#set_target_properties(hierarchical_parallelism PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(hierarchical_parallelism PRIVATE MKL::MKL_DPCPP)
target_include_directories(hierarchical_parallelism PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(hierarchical_parallelism PRIVATE)

add_executable(reduction_manual reduction_manual.cpp)
#set_target_properties(reduction_manual PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(reduction_manual PRIVATE MKL::MKL_DPCPP)
target_include_directories(reduction_manual PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(reduction_manual PRIVATE)

add_executable(oneAPI_Genz4_5D Genz4_5D.cpp)
#set_target_properties(oneAPI_Genz4_5D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(oneAPI_Genz4_5D PRIVATE MKL::MKL_DPCPP)
target_include_directories(oneAPI_Genz4_5D PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(oneAPI_Genz4_5D PRIVATE)

add_executable(oneAPI_Genz5_8D Genz5_8D.cpp)
#set_target_properties(oneAPI_Genz5_8D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(oneAPI_Genz5_8D PRIVATE MKL::MKL_DPCPP)
target_include_directories(oneAPI_Genz5_8D PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(oneAPI_Genz5_8D PRIVATE)


add_executable(oneAPI_Genz6_6D Genz6_6D.cpp)
#set_target_properties(oneAPI_Genz6_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(oneAPI_Genz6_6D PRIVATE MKL::MKL_DPCPP)
target_include_directories(oneAPI_Genz6_6D PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(oneAPI_Genz6_6D PRIVATE)

add_executable(oneAPI_Genz2_6D Genz2_6D.cpp)
#set_target_properties(oneAPI_Genz2_6D PROPERTIES POSITION_INDEPENDENT_CODE on CUDA_ARCHITECTURES ${PAGANI_TARGET_ARCH})
target_link_libraries(oneAPI_Genz2_6D PRIVATE MKL::MKL_DPCPP)
target_include_directories(oneAPI_Genz2_6D PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(oneAPI_Genz2_6D PRIVATE)