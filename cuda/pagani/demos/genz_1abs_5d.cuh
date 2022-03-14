#ifndef CUDACUHRE_TESTS_GENZ_1ABS_5D_CUH
#define CUDACUHRE_TESTS_GENZ_1ABS_5D_CUH

#include <math.h>

// From Mathematica 12.1 Integration, symbolic integration over unit hypercube.
// This is the multiplier that gives genz_1_8d an integrated value of 1 over the
// unit hypercube.

double constexpr integral = 6.371054e-01; // Value is approximate
double constexpr normalization = 1. / integral;

struct Genz_1abs_5d {

  __device__ __host__ Genz_1abs_5d(){};

  __device__ __host__ double
  operator()(double v, double w, double x, double y, double z)
  {
    return normalization * abs(cos(4. * v + 5. * w + 6. * x + 7. * y + 8. * z));
  }
};

#endif
