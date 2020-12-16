#include "function.cuh"
#include "demo_utils.cuh"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

int
main()
{
  double epsrel = 1.0e-3; // starting error tolerance.
  double const epsrel_min = 1.0e-12;
  double true_value = 1.5477367885091207413e8;
  constexpr int ndim = 6;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;
  GENZ_6_6D integrand;
  std::cout << "id, value, epsrel, epsabs, estimate, errorest, regions, "
             "converge, final, total_time\n";
  
  while (cu_time_and_call<GENZ_6_6D, ndim>("pdc_f1_latest",
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
