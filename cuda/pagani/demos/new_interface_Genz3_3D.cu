#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_3_3D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z)
  {
    return pow(1 + 3 * x + 2 * y + z, -4);
  }
};

int
main()
{

  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 3;
  GENZ_3_3D integrand;
  double true_value = 0.010846560846560846561;
  constexpr int debug = 1;
  quad::Volume<double, ndim> vol;

  while (clean_time_and_call<GENZ_3_3D, double, ndim, false, debug>(
           "f3", integrand, epsrel, true_value, "gpucuhre", std::cout, vol) ==
           true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }

  return 0;
}