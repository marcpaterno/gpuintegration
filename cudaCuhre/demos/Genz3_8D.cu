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
  double epsrel = 1e-3;//2.56000000000000067e-09;//1e-3;//5.12e-10;
  double const epsrel_min = 1.0240000000000002e-12;
  double true_value = 2.2751965817917756076e-10;
  GENZ_3_8D integrand;
  std::cout << "id, value, epsrel, epsabs, estimate, errorest, regions, fregions,"
             "converge, final, phase, total_time\n";
  constexpr int ndim = 8;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;
  
  while (cu_time_and_call<GENZ_3_8D, ndim>("Genz3_8D",
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
