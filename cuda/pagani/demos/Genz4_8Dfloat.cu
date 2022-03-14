#include "cuda/pagani/demos/demo_utils.cuh"
#include "cuda/pagani/demos/function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace detail {
  class GENZ_4_8D {
  public:
    __device__ __host__ float
    operator()(float x,
               float y,
               float z,
               float w,
               float v,
               float k,
               float m,
               float n)
    {
      float beta = .5;
      return exp(
        -1.0 * (pow(25, 2) * pow(x - beta, 2) + pow(25, 2) * pow(y - beta, 2) +
                pow(25, 2) * pow(z - beta, 2) + pow(25, 2) * pow(w - beta, 2) +
                pow(25, 2) * pow(v - beta, 2) + pow(25, 2) * pow(k - beta, 2) +
                pow(25, 2) * pow(m - beta, 2) + pow(25, 2) * pow(n - beta, 2)));
    }
  };
}

int
main()
{
  float epsrel = 1e-3;
  float const epsrel_min = 1.0240000000000002e-10;
  float true_value = (6.383802190004379e-10);
  detail::GENZ_4_8D integrand;
  PrintHeader();
  constexpr int ndim = 8;
  Config configuration;
  configuration.outfileVerbosity = 0;
  // configuration.heuristicID = 0;
  // configuration.phase_2 = true;
  while (floatIntegrands::cu_time_and_call<detail::GENZ_4_8D, ndim>(
           "8D f4",
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
