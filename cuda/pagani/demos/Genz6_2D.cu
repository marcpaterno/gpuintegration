#include "cuda/pagani/demos/demo_utils.cuh"
#include "cuda/pagani/demos/function.cuh"
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
  double true_value = 120489.75982636053604;
  GENZ_6_2D integrand;

  constexpr int ndim = 2;
  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 4;
  // configuration.phase_2 = true;

  PrintHeader();
  while (cu_time_and_call<GENZ_6_2D, ndim>("pdc_f1_latest",
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
