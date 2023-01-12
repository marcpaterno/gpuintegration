add_executable(new_interface_Genz2_6D new_interface_Genz2_6D.cpp)
target_compile_options(new_interface_Genz2_6D PRIVATE "-mllvm" "-inline-threshold=10000" "-lonemkl" "-ltbb")
#target_compile_options(new_interface_Genz2_6D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-L/opt/oneMKL/lib/" "-lonemkl" "-ltbb"  "-std=c++17")
#target_compile_options(new_interface_Genz2_6D PRIVATE "-lmkl_intel_ilp64" "-lmkl_tbb_thread" "-lmkl_core" "-lpthread"  "-lmkl_sycl" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")
#target_compile_options(new_interface_Genz2_6D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")

add_executable(new_interface_Genz3_3D new_interface_Genz3_3D.cpp)
target_compile_options(new_interface_Genz3_3D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000" "-std=c++17")

add_executable(new_interface_Genz4_5D new_interface_Genz4_5D.cpp)
target_compile_options(new_interface_Genz4_5D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000" "-std=c++17")

add_executable(new_interface_Genz5_8D new_interface_Genz5_8D.cpp)
target_compile_options(new_interface_Genz5_8D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000" "-std=c++17")

add_executable(new_interface_Genz6_6D new_interface_Genz6_6D.cpp)
target_compile_options(new_interface_Genz6_6D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000" "-std=c++17")
