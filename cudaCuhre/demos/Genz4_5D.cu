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
  double epsrel =  1e-3;//2.56000000000000067e-09;;//8.00000000000000133e-06;////
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 1.79132603674879e-06;
  GENZ_4_5D integrand;
  std::cout << "id, value, epsrel, epsabs, estimate, errorest, regions, fregions,"
             "converge, final, total_time\n";
  constexpr int ndim = 5;
    
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;
  while (cu_time_and_call<GENZ_4_5D, ndim>("Genz4_5D",
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
