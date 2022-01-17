#include "cudaPagani/demos/demo_utils.cuh"
#include "cudaPagani/demos/function.cuh"
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
  constexpr int ndim = 5;
  absCosSum5DWithoutK integrand;
  const double true_value = 0.6371054;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;

  PrintHeader();
  while (cu_time_and_call<absCosSum5DWithoutK, ndim>("absCosSum5DWithoutK",
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
