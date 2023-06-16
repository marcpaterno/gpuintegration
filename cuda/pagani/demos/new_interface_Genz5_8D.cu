#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "cuda/pagani/demos/compute_genz_integrals.cuh"
#include "common/cuda/integrands.cuh"

int
main()
{

  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 8;
  F_5_8D integrand;
  integrand.set_true_value();
  quad::Volume<double, ndim> vol;

  while (clean_time_and_call<F_5_8D, double, ndim, false>("f5",
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
