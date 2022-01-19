#include "func.cuh"
#include "quad/Cuhre.cuh"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
//#include "quad/Rule.cuh"

class Test {
public:
  __device__ __host__ double
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
    Test integrand;

    double epsrel = 1.0e-3;
    double epsabs = 1.0e-12;
    double epsrel_min = 1.0e-10;

    while (epsrel > epsrel_min) {
      Cuhre<double, 2> cuhre;
      cuhreResult res = cuhre.Integrate<Test>(integrand, epsrel, epsabs);
      printf("Genz2_2D, %.15e, %.15f, %.15f, %lu, %lu, %i\n",
             epsrel,
             res.estimate,
             res.errorest,
             res.nregions,
             res.nFinishedRegions,
             res.status);
      epsrel /= 5.0;
      printf("----------------------------------\n");
    }
  }
  Kokkos::finalize();
  return 0;
}
