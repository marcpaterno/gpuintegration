#include "cudaPagani/demos/demo_utils.cuh"
#include "cudaPagani/demos/function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace detail {
  class GENZ_6_6D {
  public:
    __device__ __host__ float
    operator()(float u, float v, float w, float x, float y, float z)
    {
      if (z > .9 || y > .8 || x > .7 || w > .6 || v > .5 || u > .4)
        return 0.;
      else
        return exp(10 * z + 9 * y + 8 * x + 7 * w + 6 * v + 5 * u);
    }
  };
}

int
main()
{
  float epsrel = 1.e-3; // starting error tolerance.
  float const epsrel_min = 1.024e-10;
  float true_value = 1.5477367885091207413e8;
  constexpr int ndim = 6;
  Config configuration;
  configuration.outfileVerbosity = 0;
  detail::GENZ_6_6D integrand;
  PrintHeader();

  while (floatIntegrands::cu_time_and_call<detail::GENZ_6_6D, ndim>(
           "6D f6",
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
