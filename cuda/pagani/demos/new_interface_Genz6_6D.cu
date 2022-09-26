#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

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

  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 6;
  GENZ_6_6D integrand;
  double true_value = 1.5477367885091207413e8;

  while (clean_time_and_call<GENZ_6_6D, ndim, false>(
           "f6", integrand, epsrel, true_value, "gpucuhre", std::cout) ==
           true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
    break;
  }

  /*epsrel = 1.0e-3;
  while (clean_time_and_call<GENZ_6_6D, ndim, true>("f6",
                                     integrand,
                                     epsrel,
                                     true_value,
                                     "gpucuhre",
                                     std::cout) == true &&
   epsrel >= epsrel_min) {
                  epsrel /= 5.0;
  }*/
  return 0;
}
