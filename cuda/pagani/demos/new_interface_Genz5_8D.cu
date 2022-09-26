#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_5_8D {
public:
  __device__ __host__ double
  operator()(double x,
             double y,
             double z,
             double k,
             double m,
             double n,
             double p,
             double q)
  {
    double beta = .5;
    double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta) -
                10. * fabs(z - beta) - 10. * fabs(k - beta) -
                10. * fabs(m - beta) - 10. * fabs(n - beta) -
                10. * fabs(p - beta) - 10. * fabs(q - beta);
    return exp(t1);
  }
};

int
main()
{

  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 8;
  GENZ_5_8D integrand;
  double true_value = 2.425217625641885e-06;

  while (clean_time_and_call<GENZ_5_8D, ndim, false>(
           "f5", integrand, epsrel, true_value, "gpucuhre", std::cout) ==
           true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }

  epsrel = 1.0e-3;
  while (clean_time_and_call<GENZ_5_8D, ndim, true>(
           "f5", integrand, epsrel, true_value, "gpucuhre", std::cout) ==
           true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }
  return 0;
}
