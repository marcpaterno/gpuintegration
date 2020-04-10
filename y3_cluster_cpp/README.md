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

1. Create a mew directory `build` in this directory.
2. cd to that directory, and run `cmake <args> ..` to generate Makefiles
3. run `make -j <N>` to build. A good value for `N` is the number of CPUs on your machine.
4. run `ctest -j <N>` to run the unit tests.

```
#!bash
cmake -DEIGEN3_INCLUDE_DIR=../../eigen -DCUBACPP_DIR=../../cubacpp \
      -DEXTERNALS_DIR=../../  \
      -DCMAKE_MODULE_PATH="../../cubacpp/cmake/modules/;../../eigen/cmake/" \
      -DCMAKE_BUILD_TYPE=<Debug|Release|RelWithDebInfo> ..
```

