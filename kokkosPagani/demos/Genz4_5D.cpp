#include "Cuhre.cuh"
#include "Rule.cuh"
#include "demo_utils.cuh"
#include "func.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class GENZ_4_5D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double w, double v)
  {
    // double alpha = 25.;
    double beta = .5;
    return exp(-1.0 *
               (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2)));
  }
};

int
main(int argc, char** argv)
{
  Kokkos::initialize();
  {
    GENZ_4_5D integrand;

    double epsrel = 1.0e-3;
    // double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;
    double true_value = 1.79132603674879e-06;
    const int ndim = 5;
    while (time_and_call<GENZ_4_5D, ndim>(
             "5D f4", integrand, epsrel, true_value, std::cout) == true &&
           epsrel >= epsrel_min) {
      epsrel /= 5.0;
      break;
    }
  }
  Kokkos::finalize();
  return 0;
}