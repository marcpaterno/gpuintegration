#include "Cuhre.cuh"
#include "Rule.cuh"
#include "demo_utils.cuh"
#include "func.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class GENZ_3_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double w,
             double v,
             double u,
             double t,
             double s)
  {
    return pow(1 + 8 * s + 7 * t + 6 * u + 5 * v + 4 * w + 3 * x + 2 * y + z,
               -9);
  }
};

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  {
    GENZ_3_8D integrand;

    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 2.2751965817917756076e-10;
    const int ndim = 8;
    while (time_and_call<GENZ_3_8D, ndim>(
             "8D f3", integrand, epsrel, true_value, std::cout) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
    }
  }
  Kokkos::finalize();
  return 0;
}