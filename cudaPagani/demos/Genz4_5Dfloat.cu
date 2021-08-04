#include "demo_utils.cuh"
#include "function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

namespace detail {
  class GENZ_4_5D {
  public:
    __device__ __host__ float
    operator()(float x, float y, float z, float w, float v)
    {
      float beta = .5;
      return exp(
        -1.0 * (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2)));
    }
  };
}

int
main()
{
  float epsrel = 1.e-3;
  float const epsrel_min = 1.0240000000000002e-10;
  float true_value = 1.79132603674879e-06;
  detail::GENZ_4_5D integrand;
  PrintHeader();
  constexpr int ndim = 5;
  Config configuration;
  configuration.outfileVerbosity = 0;
  // configuration.heuristicID = 0;
  // configuration.phase_2 = false;
  while (floatIntegrands::cu_time_and_call<detail::GENZ_4_5D, ndim>(
           "5D f4",
           integrand,
           epsrel,
           true_value,
           "gpucuhre",
           std::cout,
           configuration) == true &&
         epsrel > epsrel_min) {
    epsrel /= 5.0;
    break;
  }
}
