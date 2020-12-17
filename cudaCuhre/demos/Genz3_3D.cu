#include "function.cuh"
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
  double const epsrel_min = 1.0e-12;
  double true_value = 0.010846560846560846561;
  GENZ_3_3D integrand;
 
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;
  constexpr int ndim = 3;
  
  PrintHeader();
  while (cu_time_and_call<GENZ_3_3D, ndim>("pdc_f1_latest",
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
