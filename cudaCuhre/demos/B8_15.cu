#include "demos/function.cuh"
#include "demo_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace quad;

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double true_value = 8879.851175413485;
  double const epsrel_min = 1.0240000000000002e-10;
  BoxIntegral8_15 integrand;
  constexpr int ndim = 8;
  
  constexpr int alternative_phase1 = 0;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;
  
  while (cu_time_and_call<BoxIntegral8_15, ndim>("B8_15",
                       integrand,
                       epsrel,
                       true_value,
                       "gpucuhre",
                       std::cout,
                       configuration) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }
}
