#ifndef CUDACUHRE_TESTS_FUN6_CUH
#define CUDACUHRE_TESTS_FUN6_CUH

#include <math.h>

// From Mathematica 12.1 Integrate, symbolic integration over unit hypercube.
// This is the multilplier that gives fun6 an integrated value of 1 over the
// unit hypercube.
static double const fun6_normalization = 12.0/(7.0 - 6 * std::log(2.0) * std::log(2.0) + std::log(64.0));

double fun6(double u, double v, double w, double x, double y, double z)
{
  return fun6_normalization * (u * v + (std::pow(w, y) * x * y)/(1+u) + z*z);
}

#endif
