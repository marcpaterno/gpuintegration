add_executable(new_interface_Genz2_6D new_interface_Genz2_6D.cpp)
target_compile_options(new_interface_Genz2_6D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")

add_executable(new_interface_Genz3_3D new_interface_Genz3_3D.cpp)
target_compile_options(new_interface_Genz3_3D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")

add_executable(new_interface_Genz4_5D new_interface_Genz4_5D.cpp)
target_compile_options(new_interface_Genz4_5D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")

add_executable(new_interface_Genz5_8D new_interface_Genz5_8D.cpp)
target_compile_options(new_interface_Genz5_8D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")

add_executable(new_interface_Genz6_6D new_interface_Genz6_6D.cpp)
target_compile_options(new_interface_Genz6_6D PRIVATE "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda" "-lonemkl" "-ltbb" "-mllvm" "-inline-threshold=10000")


