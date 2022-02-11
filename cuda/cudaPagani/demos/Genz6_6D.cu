#include "cuda/cudaPagani/demos/demo_utils.cuh"
#include "cuda/cudaPagani/demos/function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace detail {
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
}

int
main()
{
  double epsrel = 1.e-3; // starting error tolerance.
  double const epsrel_min = 1.024e-10;
  double true_value = 1.5477367885091207413e8;
  constexpr int ndim = 6;
  Config configuration;
  configuration.outfileVerbosity = 0;
  // configuration.heuristicID = 0;
  // configuration.phase_2 = true;
  detail::GENZ_6_6D integrand;
  PrintHeader();

  while (cu_time_and_call_100<detail::GENZ_6_6D, ndim>("6D f6",
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
