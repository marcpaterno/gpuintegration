#include "demo_utils.cuh"
#include "function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

namespace detail {
  class GENZ_5_2D {
  public:
    __device__ __host__ double
    operator()(double x, double y)
    {
      double beta = .5;
      double t1 = -10. * fabs(x - beta) - 10. * fabs(y - beta);
      return exp(t1);
    }
  };
}

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double const epsrel_min = 1.024e-10;
  double true_value = 0.039462780237263662026;
  detail::GENZ_5_2D integrand;

  constexpr int ndim = 2;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 0;
  // configuration.phase_2 = true;
  PrintHeader();
  while (cu_time_and_call<detail::GENZ_5_2D, ndim>("GENZ_5_2D",
                                                   integrand,
                                                   epsrel,
                                                   true_value,
                                                   "gpucuhre",
                                                   std::cout,
                                                   configuration) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
}
