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
  double epsrel = 1.0e-3;
  // double epsabs = 1.0e-12;
  double const epsrel_min = 1.0240000000000002e-10;
  constexpr int ndim = 8;
  GENZ_1_8D integrand;

  Config configuration;
  configuration.outfileVerbosity = 0;
  configuration.heuristicID = 1;
  // configuration.phase_2 = true;
  double true_value =
    (1. / 315.) * sin(1.) * sin(3. / 2.) * sin(2.) * sin(5. / 2.) * sin(3.) *
    sin(7. / 2.) * sin(4.) *
    (sin(37. / 2.) - sin(35. / 2.)); /*0.000041433844333568199264*/
  ;

  PrintHeader();
  while (cu_time_and_call_100<GENZ_1_8D, ndim>("8D f1",
                                               integrand,
                                               epsrel,
                                               true_value,
                                               "gpucuhre",
                                               std::cout,
                                               configuration) == true &&
         epsrel >= epsrel_min) {
    epsrel /= 5.0;
  }

  return 0;
}
