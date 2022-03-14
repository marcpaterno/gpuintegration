#include "cuda/pagani/demos/demo_utils.cuh"
#include "cuda/pagani/demos/genz_1abs_5d.cuh"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

int
main()
{
  Genz_1abs_5d integrand;
  double epsrel = 1.0e-3;
  double const epsrel_min = 1.0240000000000002e-10;
  double true_value = 6.371054e-01; // this value is an approximation
  constexpr int ndim = 5;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 0;

  PrintHeader();
  while (cu_time_and_call<Genz_1abs_5d, ndim>("Genz_1abs_5d",
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
