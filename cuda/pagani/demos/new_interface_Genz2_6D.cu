#include <iostream>
#include "cuda/pagani/demos/new_time_and_call.cuh"
#include "cuda/pagani/demos/compute_genz_integrals.cuh"
#include "common/cuda/integrands.cuh"

int
main()
{

  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 6;
  F_2_6D integrand;
  integrand.set_true_value();
  constexpr bool use_custom = false;
  constexpr int debug = 0;
  quad::Volume<double, ndim> vol;
  bool relerr_classification = true;

  while (clean_time_and_call<F_2_6D, double, ndim, use_custom, debug>(
           "f2",
           integrand,
           epsrel,
           integrand.true_value,
           "gpucuhre",
           std::cout,
           vol,
           relerr_classification) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
    break;
  }

  return 0;
}
