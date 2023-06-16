#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "common/cuda/integrands.cuh"

class GENZ_4_6D {
public:
  __device__ __host__ double
  operator()(double x, double y, double z, double w, double v, double b)
  {
    // double alpha = 25.;
    double beta = .5;
    return exp(-1.0 * pow(25, 2) *
               (pow(x - beta, 2) + pow(y - beta, 2) + pow(z - beta, 2) +
                pow(w - beta, 2) + pow(v - beta, 2) + pow(b - beta, 2)));
  }
};

int
main()
{

  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 6;
  F_4_6D integrand;
  integrand.set_true_value();
  quad::Volume<double, ndim> vol;

  while (clean_time_and_call<F_4_6D, double, ndim, false>("f4",
                                                          integrand,
                                                          epsrel,
                                                          integrand.true_value,
                                                          "gpucuhre",
                                                          std::cout,
                                                          vol) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }

  return 0;
}
