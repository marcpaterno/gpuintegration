#include "cuda/pagani/demos/demo_utils.cuh"
#include "cuda/pagani/demos/function.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

class BoxIntegral2_12 {
public:
  __device__ __host__ double
  operator()(double x, double y)
  {
    double s = 12;
    double sum = 0;
    sum = pow(x, 2) + pow(y, 2);
    return pow(sum, s / 2);
  }
};

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double true_value = 1.592312449455;
  double const epsrel_min = 1.0240000000000002e-10;
  BoxIntegral2_12 integrand;
  constexpr int ndim = 2;

  Config configuration;
  configuration.outfileVerbosity = 0;

  PrintHeader();
  while (cu_time_and_call<BoxIntegral2_12, ndim>("B2_12",
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
