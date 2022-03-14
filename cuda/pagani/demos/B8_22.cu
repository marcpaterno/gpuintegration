#include "cuda/pagani/demos/demo_utils.cuh"
#include "cuda/pagani/demos/function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

namespace detail {
  class BoxIntegral8_22 {
  public:
    __device__ __host__ double
    operator()(double x,
               double y,
               double z,
               double k,
               double l,
               double m,
               double n,
               double o)
    {
      double s = 22;
      double sum = 0;
      sum = pow(x, 2) + pow(y, 2) + pow(z, 2) + pow(k, 2) + pow(l, 2) +
            pow(m, 2) + pow(n, 2) + pow(o, 2);
      return pow(sum, s / 2);
    }
  };
}

int
main()
{
  double epsrel = 1.e-3; // starting error tolerance.
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 1495369.283757217694;
  constexpr int ndim = 8;
  detail::BoxIntegral8_22 integrand;

  Config configuration;
  configuration.outfileVerbosity = 0;
  // configuration.heuristicID = 0;
  // configuration.phase_2 = true;
  PrintHeader();
  while (cu_time_and_call<detail::BoxIntegral8_22, ndim>("8D f7",
                                                         integrand,
                                                         epsrel,
                                                         true_value,
                                                         "gpucuhre",
                                                         std::cout,
                                                         configuration) ==
           true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
  }
}
