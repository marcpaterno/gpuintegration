#include "kokkos/pagani/demos/demo_utils.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class BoxIntegral8_22 {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double l,
             double m,
             double n,
             double o)
  {
    double s = 22;
    double sum = 0;
    sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
          pow(m, 2) + pow(n, 2) + pow(o, 2);
    return pow(sum, s / 2);
  }
};

int
main()
{
  Kokkos::initialize();
  {
    BoxIntegral8_22 integrand;
    constexpr bool use_custom = true;
    constexpr int debug = 0;
    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 1495369.283757217694;
    const int ndim = 8;
    while (time_and_call<BoxIntegral8_22, ndim, use_custom, debug>(
             "B8_22", integrand, epsrel, true_value) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
    }
  }
  Kokkos::finalize();
  return 0;
}