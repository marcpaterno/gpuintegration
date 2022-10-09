#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"

class GENZ_2_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double k, double l, double m)
  {
    double a = 50.;
    double b = .5;

    double term_1 = 1. / ((1. / pow(a, 2)) + pow(x - b, 2));
    double term_2 = 1. / ((1. / pow(a, 2)) + pow(y - b, 2));
    double term_3 = 1. / ((1. / pow(a, 2)) + pow(z - b, 2));
    double term_4 = 1. / ((1. / pow(a, 2)) + pow(k - b, 2));
    double term_5 = 1. / ((1. / pow(a, 2)) + pow(l - b, 2));
    double term_6 = 1. / ((1. / pow(a, 2)) + pow(m - b, 2));

    double val = term_1 * term_2 * term_3 * term_4 * term_5 * term_6;
    return val;
  }
};

int
main()
{

  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 6;
  GENZ_2_6D integrand;
  double true_value = 1.286889807581113e+13;
  constexpr bool use_custom_false = false;
  constexpr bool debug = false;
  quad::Volume<double, ndim> vol;

  while (clean_time_and_call<GENZ_2_6D, double, ndim, use_custom_false, debug>(
           "f2", integrand, epsrel, true_value, "gpucuhre", std::cout, vol) ==
           true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }

  return 0;
}
