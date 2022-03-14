#include "cuda/pagani/demos/demo_utils.cuh"
#include "cuda/pagani/demos/function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

namespace detail {
  class GENZ_3_3D {
  public:
    __device__ __host__ double
    operator()(double x, double y, double z)
    {
      return pow(1 + 3 * x + 2 * y + z, -4);
    }
  };
}

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double const epsrel_min = 1.024e-10;
  double true_value = 0.010846560846560846561;
  detail::GENZ_3_3D integrand;

  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 0;
  constexpr int ndim = 3;

  PrintHeader();
  while (cu_time_and_call_100<detail::GENZ_3_3D, ndim>("3D f3",
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
