This directory contains code originally from the `y3_cluster_cpp` repository
used by DES collaborators (at https://bitbucket.org/mpaterno/y3_cluster_cpp).
It has been modified to meet the needs of this project,
while retaining the essential functionality.
Only a few integrands and models from that repository have been ported.
The following are the important modifications:

1. All models have been modified to support streaming to an `ostream`, using
   hexidecimal floating point representation, to allow loading known states.

2. Interpolation classes have also been modified as described in (1).

3. The dependence on CosmoSIS has been removed by introducing a minimal set
   of mock classes with just enough functionality to allow the models and
   integrands to work.

This was taken from commit ac7697a671606fcf8d30ee45fedfea2a383b320a of
`y3_cluster_cpp`.

## Building

*TODO*: include instructions for building the `CUBA` library appropriately for
this project.

The recommended build is out-of-source.

In order for the tests to find the files it must read, you must define the environment variable `Y3_CLUSTER_CPP_DIR`
as the full path to the `y3_cluster_cpp` directory.

1. Create a mew directory `build` in this directory.
2. cd to that directory, and run `cmake <args> ..` to generate Makefiles
3. run `make -j <N>` to build. A good value for `N` is the number of CPUs on your machine.
4. run `ctest -j <N>` to run the unit tests.

The command below shows the `cmake` command to get an optimized build (`-DCMAKE_BUILD_TYPE=Release`).
Other recognized values are `Debug` and `DebWithRelInfo`.

```
cmake -DEIGEN3_INCLUDE_DIR=../../eigen -DCUBACPP_DIR=../../cubacpp -DEXTERNALS_DIR=../../  -DCMAKE_MODULE_PATH="../../cubacpp/cmake/modules/;../../eigen/cmake/" -DCMAKE_BUILD_TYPE=Release ..
```

It may be convenient to define `PROJECT_DIR` to be the top-level directory for this repository.

```
cmake -DCUBA_INCLUDE_DIR=${PROJECT_DIR}/cuba \
      -DCUBA_LIBRARIES=${PROJECT_DIR}/cuba/libcuba.so \
      -DEIGEN3_INCLUDE_DIR=${PROJECT_DIR}/eigen \
      -DCUBACPP_DIR=${PROJECT_DIR}/cubacpp \
      -DEXTERNALS_DIR=${PROJECT_DIR} \
      -DCMAKE_MODULE_PATH="${PROJECT_DIR}/cubacpp/cmake/modules/;${PROJECT_DIR}/eigen/cmake/" \
      -DCMAKE_BUILD_TYPE=Release ..
```

