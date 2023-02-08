#include "kokkos/pagani/demos/demo_utils.cuh"
#include "func.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class Genz2_2D {
public:
  KOKKOS_INLINE_FUNCTION double
  operator()(double x, double y)
  {
    double a = 50.;
    double b = .5;

    double term_1 = 1. / ((1. / pow(a, 2)) + pow(x - b, 2));
    double term_2 = 1. / ((1. / pow(a, 2)) + pow(y - b, 2));

    double val = term_1 * term_2;
    return val;
  }
};

int
main()
{
  Kokkos::initialize();
  {
    Genz2_2D integrand;
    constexpr bool use_custom = true;
    constexpr int debug = 0;
    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 23434.04;
    const int ndim = 2;
    while (time_and_call<Genz2_2D, ndim, use_custom, debug>(
             "2D f2", integrand, epsrel, true_value) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
    }
  }

  Kokkos::finalize();
  return 0;
}