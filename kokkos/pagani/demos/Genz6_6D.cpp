#include "kokkos/pagani/quad/Cuhre.cuh"
#include "kokkos/pagani/quad/Rule.cuh"
#include "kokkos/pagani/demos/demo_utils.cuh"
#include "kokkos/pagani/quad/func.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class GENZ_6_6D {
public:
  __device__ __host__ double
  operator()(double u, double v, double w, double x, double y, double z)
  {
    if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
      return 0.;
    else
      return exp(10 * z + 9 * y + 8 * x + 7 * w + 6 * v +
                 5 * u) /*/1.5477367885091207413e8*/;
  }
};

int
main()
{
  Kokkos::initialize();
  {
    GENZ_6_6D integrand;

    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 1.5477367885091207413e8;
    const int ndim = 6;
    while (time_and_call<GENZ_6_6D, ndim>(
             "6D f6", integrand, epsrel, true_value, std::cout) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
    }
  }
  Kokkos::finalize();
  return 0;
}