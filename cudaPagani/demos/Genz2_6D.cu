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
  double epsrel = 1.e-3; // starting error tolerance.
  double const epsrel_min = 1.024e-10;
  double true_value = 1.286889807581113e+13;
  GENZ_2_6D integrand;
  constexpr int ndim = 6;
  Config configuration;
  configuration.outfileVerbosity = 0;
  // configuration.heuristicID = 0;

  PrintHeader();
  while (cu_time_and_call<GENZ_2_6D, ndim>("GENZ2_6D",
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
